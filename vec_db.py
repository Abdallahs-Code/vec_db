from typing import Annotated
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import math
import json
import time
from pathlib import Path

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
DB_SEED_NUMBER = 42

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        # Initialize database paths
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # Build index for database
            self._build_index()

    def _get_num_records(self) -> int:
        # Get number of records in the database
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]) -> None:
        # Insert new records into the database
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        # Rebuild index after insertion
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # Get a single row from the database
        num_records = self._get_num_records()
        if row_num < 0 or row_num >= num_records:
            raise ValueError(f"Invalid row number: {row_num}. Must be between 0 and {num_records - 1}.")
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

    def get_all_rows(self) -> np.ndarray:
        # Get all rows from the database
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _cal_score(self, vec1, vec2):
        # Calculate cosine similarity score between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def compute_clustering_parameters(self):
        db_size = self._get_num_records()
        # Sample size: 5% of db, good enough for all db sizes since the lowest db size is 1M
        self.sample_size = int(0.05 * db_size)
        # Number of clusters: nearest power of 2 of sqrt(db_size) -> rule of thumb
        self.n_clusters = 2 ** int(round(math.log2(math.sqrt(db_size))))
        # MiniBatch batch size is 4096 which is good for db sizes 1M - 20M
        self.batch_size = 4096
        # Max iterations appropriate regardless of db size
        self.max_iter = 200

    def sample_for_kmeans(self, seed : int = DB_SEED_NUMBER) -> np.ndarray:
        # Sample vectors from the database for k-means training
        sample_size = self.sample_size
        num_records = self._get_num_records()
        if sample_size > num_records:
            raise ValueError("Sample size cannot be greater than the number of records in the database.")
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(num_records, sample_size, replace=False)
        samples = np.zeros((sample_size, DIMENSION), dtype=np.float32)
        for i, index in enumerate(sample_indices):
            samples[i] = self.get_one_row(index)
        return samples

    def train_centroids(self, sampled_vectors: np.ndarray) -> np.ndarray:
        # Train k-means centroids using MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=42
        )
        kmeans.fit(sampled_vectors)
        return kmeans.cluster_centers_.astype(np.float32)
    
    def save_centroids(self, centroids: np.ndarray):
        # Ensure centroids are float32
        if centroids.dtype != np.float32:
            centroids = centroids.astype(np.float32)
        # Save centroids to disk
        with open(self.index_path, 'wb') as f:
            centroids.tofile(f)
        # Validate with file size
        expected_size = centroids.shape[0] * centroids.shape[1] * ELEMENT_SIZE
        actual_size = os.path.getsize(self.index_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Centroid file size mismatch. Expected {expected_size} bytes, got {actual_size} bytes"
            )

    def load_centroids(self, n_clusters: int) -> np.ndarray:
        # If centroids file does not exist, raise error
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Centroids file not found: {self.index_path}")
        # Validate with file size
        expected_size = n_clusters * DIMENSION * ELEMENT_SIZE
        actual_size = os.path.getsize(self.index_path)
        if actual_size != expected_size:
            raise ValueError(
                f"Centroids file size mismatch. Expected {expected_size} bytes "
                f"for {n_clusters} clusters, but got {actual_size} bytes. "
                f"File may be corrupted or n_clusters is incorrect."
            )
        # Load centroids from disk
        with open(self.index_path, 'rb') as f:
            centroids = np.fromfile(f, dtype=np.float32, count=n_clusters * DIMENSION)
        # Reshape to (n_clusters, DIMENSION)
        return centroids.reshape(n_clusters, DIMENSION)

    def _build_index(self):
        """
        Build IVF index with robust error handling and efficient memory usage.

        Steps:
        1. Compute clustering params
        2. Check if existing index is valid, rebuild if not
        3. Train OR load centroids (with proper caching)
        4. Assign vectors to clusters
        5. Build inverted index files
        6. Save metadata
        """

        # ----------------------------------------------------
        # 1. Compute clustering parameters for current DB
        # ----------------------------------------------------
        self.compute_clustering_parameters()

        # ----------------------------------------------------
        # 2. Check if existing index is valid for current DB
        # ----------------------------------------------------
        need_rebuild = self._should_rebuild_index()

        if need_rebuild:
            print("Index invalid or outdated. Rebuilding...")
            self._cleanup_old_index()

        # ----------------------------------------------------
        # 3. Get centroids (train if needed, otherwise load)
        # ----------------------------------------------------
        centroids = self._get_or_create_centroids()

        # ----------------------------------------------------
        # 4. Prepare index directory
        # ----------------------------------------------------
        self.index_dir = str(Path(self.index_path).with_suffix("")) + "_idx"
        os.makedirs(self.index_dir, exist_ok=True)

        # Clean up old cluster files
        for old_file in Path(self.index_dir).glob("cluster_*.ids"):
            try:
                old_file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {old_file}: {e}")

        # ----------------------------------------------------
        # 5. Prepare for vector assignment
        # ----------------------------------------------------
        num_records = self._get_num_records()

        # Calculate optimal chunk size
        bytes_per_vector = DIMENSION * ELEMENT_SIZE
        target_chunk_bytes = 16 * 1024 * 1024  # 16MB chunks
        chunk_size = max(1, min(num_records, target_chunk_bytes // bytes_per_vector))
        chunk_size = min(chunk_size, 65536)  # Cap at 64K vectors per chunk

        # Create empty cluster files
        cluster_paths = []
        for ci in range(self.n_clusters):
            cluster_file = os.path.join(self.index_dir, f"cluster_{ci}.ids")
            # Ensure file exists and is empty
            with open(cluster_file, 'wb') as f:
                pass  # Create empty file
            cluster_paths.append(cluster_file)

        # Initialize cluster counts
        cluster_counts = [0] * self.n_clusters

        # ----------------------------------------------------
        # 6. Assign vectors to clusters in chunks
        # ----------------------------------------------------
        try:
            # Open memmap for reading database
            db_vectors = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                shape=(num_records, DIMENSION)
            )

            # Process in chunks to manage memory
            for chunk_start in range(0, num_records, chunk_size):
                chunk_end = min(num_records, chunk_start + chunk_size)

                # Load chunk into memory
                chunk_vectors = np.array(db_vectors[chunk_start:chunk_end], dtype=np.float32)

                # Normalize vectors in chunk
                chunk_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
                chunk_norms[chunk_norms == 0] = 1.0
                chunk_vectors_normalized = chunk_vectors / chunk_norms

                # Compute similarities to centroids (cosine similarity via dot product)
                similarities = np.dot(chunk_vectors_normalized, centroids.T)

                # Assign each vector to nearest centroid
                cluster_assignments = np.argmax(similarities, axis=1)

                # Write vector IDs to appropriate cluster files
                for cluster_id in range(self.n_clusters):
                    # Find vectors assigned to this cluster
                    mask = (cluster_assignments == cluster_id)
                    local_indices = np.where(mask)[0]

                    if local_indices.size == 0:
                        continue

                    # Convert to global IDs
                    global_ids = (chunk_start + local_indices).astype(np.uint32)

                    # Append to cluster file
                    with open(cluster_paths[cluster_id], 'ab') as f:
                        global_ids.tofile(f)

                    cluster_counts[cluster_id] += len(global_ids)

            # Clean up memmap
            del db_vectors

        except Exception as e:
            raise RuntimeError(f"Error during vector assignment: {e}")

        # ----------------------------------------------------
        # 7. Save metadata
        # ----------------------------------------------------
        metadata = {
            "n_clusters": int(self.n_clusters),
            "num_records": int(num_records),
            "dimension": int(DIMENSION),
            "cluster_files": {
                str(i): os.path.basename(cluster_paths[i])
                for i in range(self.n_clusters)
            },
            "counts": {
                str(i): int(cluster_counts[i])
                for i in range(self.n_clusters)
            },
            "centroids_file": os.path.basename(self.index_path),
            "build_timestamp": time.time()
        }

        metadata_path = os.path.join(self.index_dir, "index_meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Verify all cluster files exist
        for cluster_id, cluster_path in enumerate(cluster_paths):
            if not os.path.exists(cluster_path):
                raise RuntimeError(f"Cluster file {cluster_path} was not created")

    def _should_rebuild_index(self) -> bool:
        """
        Check if the existing index is valid for the current database.
        Returns True if index needs to be rebuilt.
        """
        # If index file doesn't exist, need to build
        if not os.path.exists(self.index_path):
            return True

        # If index file is empty, need to build
        if os.path.getsize(self.index_path) == 0:
            return True

        # Check if metadata exists
        self.index_dir = str(Path(self.index_path).with_suffix("")) + "_idx"
        metadata_path = os.path.join(self.index_dir, "index_meta.json")

        if not os.path.exists(metadata_path):
            return True

        try:
            # Load existing metadata
            with open(metadata_path, 'r') as f:
                old_metadata = json.load(f)

            # Check if n_clusters matches
            old_n_clusters = int(old_metadata.get("n_clusters", 0))
            if old_n_clusters != self.n_clusters:
                print(f"n_clusters mismatch: old={old_n_clusters}, new={self.n_clusters}")
                return True

            # Check if num_records matches
            old_num_records = int(old_metadata.get("num_records", 0))
            current_num_records = self._get_num_records()
            if old_num_records != current_num_records:
                print(f"num_records mismatch: old={old_num_records}, new={current_num_records}")
                return True

            # Check if centroids file size is correct
            expected_centroid_size = old_n_clusters * DIMENSION * ELEMENT_SIZE
            actual_centroid_size = os.path.getsize(self.index_path)
            if expected_centroid_size != actual_centroid_size:
                print(f"Centroid file corrupted: expected={expected_centroid_size}, actual={actual_centroid_size}")
                return True

            # Index appears valid
            return False

        except Exception as e:
            print(f"Error reading metadata: {e}")
            return True


    def _cleanup_old_index(self):
        """
        Remove old/invalid index files.
        """
        # Delete centroids file
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                print(f"Removed old centroids file: {self.index_path}")
            except Exception as e:
                print(f"Warning: Could not remove {self.index_path}: {e}")

        # Delete index directory
        self.index_dir = str(Path(self.index_path).with_suffix("")) + "_idx"
        if os.path.exists(self.index_dir):
            try:
                import shutil
                shutil.rmtree(self.index_dir)
                print(f"Removed old index directory: {self.index_dir}")
            except Exception as e:
                print(f"Warning: Could not remove {self.index_dir}: {e}")


    def _get_or_create_centroids(self) -> np.ndarray:
        """
        Get centroids either by training or loading from disk.
        This method handles all the logic for centroid management.

        Returns:
            np.ndarray: Normalized centroids of shape (n_clusters, DIMENSION)
        """
        need_training = (
            not os.path.exists(self.index_path) or
            os.path.getsize(self.index_path) == 0
        )

        if need_training:
            # Train new centroids
            print(f"Training {self.n_clusters} centroids...")

            # Sample vectors for training
            sampled_vectors = self.sample_for_kmeans()

            # Adjust n_clusters if we don't have enough samples
            if sampled_vectors.shape[0] < self.n_clusters:
                old_n = self.n_clusters
                self.n_clusters = sampled_vectors.shape[0]
                print(f"Warning: Adjusted n_clusters from {old_n} to {self.n_clusters} due to sample size")

            # Train k-means
            centroids = self.train_centroids(sampled_vectors)

            # Normalize centroids
            centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroid_norms[centroid_norms == 0] = 1.0
            centroids = (centroids / centroid_norms).astype(np.float32)

            # Save to disk
            self.save_centroids(centroids)
            print(f"Centroids trained and saved to {self.index_path}")

            return centroids
        else:
            # Load existing centroids
            print(f"Loading existing centroids from {self.index_path}")
            centroids = self.load_centroids(self.n_clusters)

            # Normalize to ensure correctness
            centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroid_norms[centroid_norms == 0] = 1.0
            centroids = (centroids / centroid_norms).astype(np.float32)

            return centroids

    def load_inverted_list(self, cluster_id: int) -> np.ndarray:
        # Ensure index_dir exists
        if not hasattr(self, "index_dir"):
            self.index_dir = str(Path(self.index_path).with_suffix('')) + "_idx"

        # Load metadata
        meta_path = os.path.join(self.index_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        file_name = meta["cluster_files"].get(str(cluster_id))
        if file_name is None:
            return np.memmap(None, dtype=np.uint32, mode="r", shape=(0,))
        file_path = os.path.join(self.index_dir, file_name)
        if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
            return np.memmap(None, dtype=np.uint32, mode="r", shape=(0,))
        count = os.path.getsize(file_path) // np.dtype(np.uint32).itemsize

        # Memory-map the file WITHOUT reading it into RAM
        return np.memmap(file_path, dtype=np.uint32, mode="r", shape=(count,))


    def get_index_metadata(self) -> dict:
        """Return index metadata as a dict (reads index_meta.json)."""
        if not hasattr(self, "index_dir"):
            self.index_dir = str(Path(self.index_path).with_suffix('')) + "_idx"
        meta_path = os.path.join(self.index_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            return json.load(fh)


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        """
        Disk-based IVF retrieval:
        - normalize query
        - load centroids from disk
        - compute similarity to all centroids and pick top-nprobe
        - load inverted lists for those clusters
        - load only the vectors referenced by those lists using memmap indexing
        - compute cosine similarity and return top_k row ids (as ints)

        Returns:
        List[int] top_k ids (row numbers)
        """
        # 1) validate & normalize query
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.size != DIMENSION:
            raise ValueError(f"Query dimension mismatch: expected {DIMENSION}, got {q.size}")
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            # zero-vector query -> all similarities zero; return smallest ids
            qn = q
        else:
            qn = q / q_norm
        # 2) ensure index exists & load metadata
        if not hasattr(self, "index_dir"):
            self.index_dir = str(Path(self.index_path).with_suffix('')) + "_idx"
        meta_path = os.path.join(self.index_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)

        n_clusters = int(meta["n_clusters"])
        num_records = int(meta["num_records"])

        # 3) load centroids (allowed in RAM)
        centroids = np.array(self.load_centroids(n_clusters), dtype=np.float32)
        # normalize centroids just in case
        c_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_norm[c_norm == 0] = 1.0
        centroids = centroids / c_norm

        # 4) compute similarities to centroids and select top-nprobe
        sims_to_centroids = np.dot(centroids, qn.astype(np.float32))  # shape (n_clusters,)
        # choose nprobe heuristically depending on db size (tweakable)
        if num_records <= 1_000_000:
            nprobe = 3
        elif num_records <= 10_000_000:
            nprobe = 6
        elif num_records <= 15_000_000:
            nprobe = 7
        else:
            nprobe = 8
        nprobe = min(max(1, nprobe), n_clusters)

        # get top-nprobe centroid indices
        top_centroid_idxs = np.argpartition(-sims_to_centroids, nprobe-1)[:nprobe]
        # but sort them descending for determinism (optional)
        top_centroid_idxs = top_centroid_idxs[np.argsort(-sims_to_centroids[top_centroid_idxs])]

        # 5) gather candidate ids from inverted lists (disk memmap)
        candidate_ids_list = []
        total_candidates = 0
        for ci in top_centroid_idxs:
            ids = self.load_inverted_list(int(ci))  # returns memmap of uint32 (or empty memmap)
            if ids.size == 0:
                continue
            # convert memmap view to ndarray of dtype uint32 (this doesn't read DB vectors)
            ids_arr = np.array(ids, dtype=np.uint32)
            candidate_ids_list.append(ids_arr)
            total_candidates += ids_arr.size

        if total_candidates == 0:
            return []

        # concat and deduplicate candidate ids (dedupe helps if vectors assigned to multiple probed clusters)
        candidate_ids = np.concatenate(candidate_ids_list, axis=0)
        # stable unique while keeping numeric order: use np.unique which sorts; that's OK
        candidate_ids = np.unique(candidate_ids.astype(np.uint32))

        # If candidate set is huge, optionally limit (safety guard). You can adjust or remove.
        MAX_CANDIDATES = 200_000  # heuristic cap to avoid blowing memory; tune as needed
        if candidate_ids.size > MAX_CANDIDATES:
            # keep top candidates by centroid-sim score: approximate by mapping each id to its cluster sim
            # simpler: take random subset but deterministic: take the smallest ids first (deterministic)
            candidate_ids = np.sort(candidate_ids)[:MAX_CANDIDATES]

        # 6) Read only required vectors from DB using memmap advanced indexing
        # Note: this will allocate candidate_vectors in RAM (size = n_candidates * DIMENSION * 4 bytes)
        data = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        try:
            candidate_vectors = np.array(data[candidate_ids], dtype=np.float32)  # loads only needed rows
        except Exception as e:
            raise RuntimeError(f"Failed reading candidate vectors: {e}")

        # 7) normalize candidate vectors
        norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        candidate_vectors = candidate_vectors / norms

        # 8) compute cosine similarity scores with query (dot of normalized vectors)
        scores = candidate_vectors.dot(qn.astype(np.float32))  # shape (n_candidates,)

        # 9) rank results: primary key = -score, secondary = id (smaller id wins ties)
        # We'll use lexsort: keys are (ids, -scores) so lexsort((ids, -scores)) gives ascending by -score then ids
        neg_scores = -scores
        # convert candidate_ids to signed ints for lexsort stability
        ids_for_sort = candidate_ids.astype(np.int64)
        order = np.lexsort((ids_for_sort, neg_scores))
        # pick top_k
        topk_idx = order[:top_k]
        topk_ids = candidate_ids[topk_idx].tolist()

        return [int(x) for x in topk_ids]
