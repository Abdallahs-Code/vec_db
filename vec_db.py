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
        # Check for database file existence
        if not os.path.isfile(database_file_path):
            print("Database file does NOT exist.")
        # Initialize database paths
        self.db_path = database_file_path
        self.index_path = index_file_path
        # Build index
        self._build_index()

    def _get_num_records(self) -> int:
        # Get number of records in the database
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]) -> None:
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        new_total_records = num_old_records + num_new_records
        # Ensure database file is large enough
        with open(self.db_path, 'ab') as f:
            f.truncate(new_total_records * DIMENSION * ELEMENT_SIZE)
        # Memory-map the full file for writing
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=(new_total_records, DIMENSION))
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index(inserted=True)

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
        # Randomly select indices
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(num_records, sample_size, replace=False)
        # Use a single memmap to read all rows at once
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        samples = db_vectors[sample_indices]
        # Return a copy as float32 array (to avoid memmap issues)
        return np.array(samples, dtype=np.float32)

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

    def _build_index(self, inserted: bool = False) -> None:
        # Compute clustering parameters
        self.compute_clustering_parameters()
        # Check if we need to rebuild index
        need_rebuild = self._should_rebuild_index()
        if need_rebuild:
            print("Index invalid or outdated. Rebuilding...")
            self._cleanup_old_index()
        else:
            if (inserted):
                print("Database updated. Rebuilding index...")
                self._cleanup_old_index()
            else:
                print("Index is up-to-date. No rebuild needed.")
                return
        # Get or create centroids
        centroids = self._get_or_create_centroids()
        # Creating  index directory for metadata and cluster files
        self.index_dir = str(Path(self.index_path).with_suffix("")) + "_idx"
        os.makedirs(self.index_dir, exist_ok=True)
        # Clean up old cluster files
        for old_file in Path(self.index_dir).glob("cluster_*.ids"):
            try:
                old_file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {old_file}: {e}")
        # Preparing for vector assignment
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
        # Assign vectors to clusters in chunks
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
        # Write metadata file
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
        # If index file doesn't exist, need to build
        if not os.path.exists(self.index_path):
            return True
        # Check if metadata exists
        metadata_path = os.path.join(str(Path(self.index_path).with_suffix("")) + "_idx", "index_meta.json")
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
        need_training = (not os.path.exists(self.index_path) or os.path.getsize(self.index_path) == 0)
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

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        # Validate & normalize query
        qn = np.asarray(query, dtype=np.float32).reshape(-1)
        if qn.size != DIMENSION:
            raise ValueError(f"Query dimension mismatch: expected {DIMENSION}, got {qn.size}")
        q_norm = np.linalg.norm(qn)
        if q_norm > 0:
            qn /= q_norm
        # Ensure index exists & load metadata
        if not hasattr(self, "index_dir"):
            self.index_dir = str(Path(self.index_path).with_suffix('')) + "_idx"
        meta_path = os.path.join(self.index_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        n_clusters = int(meta["n_clusters"])
        num_records = int(meta["num_records"])
        # Load centroids (allowed in RAM)
        centroids = self.load_centroids(n_clusters)
        # Compute similarities to centroids and select top-nprobe
        sims_to_centroids = centroids.dot(qn)
        # Choose nprobe heuristically depending on db size (tweakable)
        if num_records <= 1_000_000:
            nprobe = 3
        elif num_records <= 10_000_000:
            nprobe = 6
        elif num_records <= 20_000_000:
            nprobe = 8
        nprobe = min(max(1, nprobe), n_clusters)
        # Get top-nprobe centroid indices
        top_centroid_idxs = np.argpartition(-sims_to_centroids, nprobe-1)[:nprobe]
        # Gather candidate ids from inverted lists (disk memmap)
        candidate_ids_list = []
        for ci in top_centroid_idxs:
            ids = self.load_inverted_list(int(ci))  # Returns memmap of uint32 (or empty memmap)
            if ids.size == 0:
                continue
            candidate_ids_list.append(ids)
        if not candidate_ids_list:
            return []
        # Concat and deduplicate candidate ids (dedupe helps if vectors assigned to multiple probed clusters)
        candidate_ids = np.concatenate(candidate_ids_list, axis=0)
        # Read only required vectors from DB using memmap advanced indexing
        # Note: this will allocate candidate_vectors in RAM (size = n_candidates * DIMENSION * 4 bytes)
        data = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        try:
            candidate_vectors = np.array(data[candidate_ids], dtype=np.float32)  # Loads only needed rows
        except Exception as e:
            raise RuntimeError(f"Failed reading candidate vectors: {e}")
        # Normalize candidate vectors
        norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        candidate_vectors = candidate_vectors / norms
        # Compute cosine similarity scores with query (dot of normalized vectors)
        scores = candidate_vectors.dot(qn)  # Shape (n_candidates,)
        # If top_k is much smaller than number of candidates, use partial sort
        if top_k < candidate_ids.size:
            # Get top_k candidate indices (unordered)
            topk_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            # Sort only the top_k for deterministic results (by -score, then candidate ID)
            topk_idx = topk_idx[np.lexsort((candidate_ids[topk_idx], -scores[topk_idx]))]
        else:
            # If top_k >= n_candidates, just sort everything
            topk_idx = np.lexsort((candidate_ids, -scores))
        # Select top_k IDs
        topk_ids = candidate_ids[topk_idx].tolist()
        return [int(x) for x in topk_ids]