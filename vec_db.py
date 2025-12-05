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
PQ_M = 8
PQ_KSUB = 256
POSTING_DTYPE = np.dtype([
    ('id', np.uint32),
    ('code', np.uint8, (PQ_M,))
])

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index", new_db = True, db_size = None) -> None:
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
        clusters_file_path = os.path.join(self.index_path, "centroids.dat")
        with open(clusters_file_path, 'wb') as f:
            centroids.tofile(f)
        # Validate with file size
        expected_size = centroids.shape[0] * centroids.shape[1] * ELEMENT_SIZE
        actual_size = os.path.getsize(clusters_file_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Centroid file size mismatch. Expected {expected_size} bytes, got {actual_size} bytes"
            )

    def load_centroids(self, n_clusters: int) -> np.ndarray:
        clusters_file_path = os.path.join(self.index_path, "centroids.dat")
        # If centroids file does not exist, raise error
        if not os.path.exists(clusters_file_path):
            raise FileNotFoundError(f"Centroids file not found: {clusters_file_path}")
        # Validate with file size
        expected_size = n_clusters * DIMENSION * ELEMENT_SIZE
        actual_size = os.path.getsize(clusters_file_path)
        if actual_size != expected_size:
            raise ValueError(
                f"Centroids file size mismatch. Expected {expected_size} bytes "
                f"for {n_clusters} clusters, but got {actual_size} bytes. "
                f"File may be corrupted or n_clusters is incorrect."
            )
        # Load centroids from disk
        with open(clusters_file_path, 'rb') as f:
            centroids = np.fromfile(f, dtype=np.float32, count=n_clusters * DIMENSION)
        # Reshape to (n_clusters, DIMENSION)
        return centroids.reshape(n_clusters, DIMENSION)
    
    def train_pq_codebook(self, samples: np.ndarray) -> np.ndarray:
        """
        Train Product Quantization codebook using given samples.
        Input:
            samples: (N, D) float32
        Output:
            codebooks: (m, ksub, D/m) float32
        """
        m = PQ_M
        ksub = PQ_KSUB
        d = DIMENSION // m

        assert samples.ndim == 2 and samples.shape[1] == DIMENSION, \
            f"Expected samples of shape (N, {DIMENSION}), got {samples.shape}"

        # Split into sub-vectors
        sub_vectors = samples.reshape(-1, m, d)

        codebooks = np.zeros((m, ksub, d), dtype=np.float32)

        # Train subquantizers independently
        for si in range(m):
            sub_data = sub_vectors[:, si, :]

            kmeans = MiniBatchKMeans(
                n_clusters=ksub,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                random_state=DB_SEED_NUMBER + si
            )
            kmeans.fit(sub_data)
            codebooks[si] = kmeans.cluster_centers_.astype(np.float32)

        return codebooks
    
    def save_pq_codebook(self, codebooks: np.ndarray):
        expected_shape = (PQ_M, PQ_KSUB, DIMENSION // PQ_M)
        if codebooks.shape != expected_shape:
            raise ValueError(f"Codebook shape mismatch: expected {expected_shape}, got {codebooks.shape}")

        file_path = os.path.join(self.index_path, "pq_codebook.dat")
        with open(file_path, "wb") as f:
            codebooks.astype(np.float32).tofile(f)

        expected_size = PQ_M * PQ_KSUB * (DIMENSION // PQ_M) * ELEMENT_SIZE
        actual_size = os.path.getsize(file_path)
        if expected_size != actual_size:
            raise RuntimeError(
                f"Codebook file corrupted: expected={expected_size}, actual={actual_size}"
            )
        
    def load_pq_codebook(self) -> np.ndarray:
        file_path = os.path.join(self.index_path, "pq_codebook.dat")
        if not os.path.exists(file_path):
            raise FileNotFoundError("PQ codebook not found — build index first")

        expected_size = PQ_M * PQ_KSUB * (DIMENSION // PQ_M) * ELEMENT_SIZE
        actual_size = os.path.getsize(file_path)
        if expected_size != actual_size:
            raise RuntimeError(
                f"PQ codebook file size mismatch: expected {expected_size}, got {actual_size}"
            )

        with open(file_path, "rb") as f:
            codebooks = np.fromfile(f, dtype=np.float32)

        return codebooks.reshape(PQ_M, PQ_KSUB, DIMENSION // PQ_M)

    def _build_index(self) -> None:
        # Compute clustering parameters
        self.compute_clustering_parameters()
        # Check if we need to rebuild index
        need_rebuild = self._should_rebuild_index()
        if need_rebuild:
            print("Building new index...")
            self._cleanup_old_index()
        else:
            print("Index is up-to-date. No rebuild needed.")
            return
        # Creating index directory
        os.makedirs(self.index_path, exist_ok=True)
        # Sample vectors for centroid training
        sampled_vectors = self.sample_for_kmeans()
        # Get or create centroids
        centroids = self._get_or_create_centroids(sampled_vectors)
        # Get or create PQ codebook
        pq_codebook = self._get_or_create_pq_codebook(sampled_vectors)
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
            cluster_file = os.path.join(self.index_path, f"cluster_{ci}.ids")
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
                # Load chunk into memory (raw vectors)
                chunk_vectors = np.array(db_vectors[chunk_start:chunk_end], dtype=np.float32)  # (L, DIMENSION)

                # Normalize vectors for centroid assignment (cosine)
                chunk_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
                chunk_norms[chunk_norms == 0] = 1.0
                chunk_vectors_normalized = chunk_vectors / chunk_norms

                # Compute similarities to centroids (cosine similarity via dot product)
                similarities = np.dot(chunk_vectors_normalized, centroids.T)
                # Assign each vector to nearest centroid
                cluster_assignments = np.argmax(similarities, axis=1)

                # Encode the chunk into PQ codes (L x m uint8)
                chunk_codes = self._encode_chunk_to_pq(chunk_vectors, pq_codebook)

                # Prepare posting dtype: id:uint32 + code:uint8[m]
                posting_dtype = np.dtype([('id', np.uint32), ('code', np.uint8, (PQ_M,))])

                # Write vector IDs + codes to appropriate cluster files
                for cluster_id in range(self.n_clusters):
                    mask = (cluster_assignments == cluster_id)
                    local_indices = np.where(mask)[0]
                    if local_indices.size == 0:
                        continue

                    # Global IDs
                    global_ids = (chunk_start + local_indices).astype(np.uint32)  # (k,)

                    # Codes to write
                    codes_to_write = chunk_codes[local_indices]                  # (k, m) dtype uint8

                    # Build structured array and write to file
                    arr = np.empty(local_indices.size, dtype=posting_dtype)
                    arr['id'] = global_ids
                    arr['code'] = codes_to_write

                    # Append to cluster file
                    with open(cluster_paths[cluster_id], 'ab') as f:
                        arr.tofile(f)

                    cluster_counts[cluster_id] += local_indices.size

                # free chunk arrays
                del chunk_vectors, chunk_vectors_normalized, chunk_codes, similarities, cluster_assignments
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
            "centroids_file": "centroids.dat",
            "codebook_file": "pq_codebook.dat",  
            "pq": {
                "m": PQ_M,
                "ksub": PQ_KSUB,
                "subdim": DIMENSION // PQ_M,
                "posting_bytes": (4 + PQ_M)
            },
            "build_timestamp": time.time()
        }
        metadata_path = os.path.join(self.index_path, "index_meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        # Verify all cluster files exist
        for cluster_id, cluster_path in enumerate(cluster_paths):
            if not os.path.exists(cluster_path):
                raise RuntimeError(f"Cluster file {cluster_path} was not created")
        print("Index build complete.")

    def _should_rebuild_index(self) -> bool:
        # If index file doesn't exist, need to build
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        if not os.path.exists(centroids_path):
            return True
        # Check if metadata exists
        metadata_path = os.path.join(self.index_path, "index_meta.json")
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
            actual_centroid_size = os.path.getsize(centroids_path)
            if expected_centroid_size != actual_centroid_size:
                print(f"Centroid file corrupted: expected={expected_centroid_size}, actual={actual_centroid_size}")
                return True
            # Index appears valid
            return False
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return True

    def _cleanup_old_index(self):
        # Delete old index directory if it exists
        if os.path.isdir(self.index_path):
            try:
                import shutil
                shutil.rmtree(self.index_path)
                print(f"Removed old index directory: {self.index_path}")
            except Exception as e:
                print(f"Warning: Could not remove {self.index_path}: {e}")

    def _get_or_create_centroids(self, sampled_vectors: np.ndarray) -> np.ndarray:
        centroids_file = os.path.join(self.index_path, "centroids.dat")
        need_training = (not os.path.exists(centroids_file) or os.path.getsize(centroids_file) == 0)
        if need_training:
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
        
    def _get_or_create_pq_codebook(self, samples: np.ndarray) -> np.ndarray:
        """
        Load existing PQ codebook or train a new one using provided samples.
        """
        pq_file = os.path.join(self.index_path, "pq_codebook.dat")

        # 1️⃣ Load if exists
        if os.path.exists(pq_file):
            return self.load_pq_codebook()

        # 2️⃣ Otherwise train & save
        codebooks = self.train_pq_codebook(samples)
        self.save_pq_codebook(codebooks)
        return codebooks
    
    def _encode_chunk_to_pq(self, chunk_vectors: np.ndarray, pq_codebook: np.ndarray) -> np.ndarray:
        """
        Encode chunk_vectors (shape: (N, DIMENSION)) into PQ codes using pq_codebook.
        pq_codebook shape: (m, ksub, subdim)  (float32)
        Returns: codes array shape (N, m) dtype=np.uint8
        Uses squared L2 assignment per subspace.
        """
        if chunk_vectors.size == 0:
            return np.zeros((0, PQ_M), dtype=np.uint8)

        m = PQ_M
        subdim = DIMENSION // m
        ksub = PQ_KSUB

        # Ensure shapes
        assert pq_codebook.shape == (m, ksub, subdim), f"pq_codebook shape mismatch: {pq_codebook.shape}"

        # Reshape chunk to (N, m, subdim)
        N = chunk_vectors.shape[0]
        subs = chunk_vectors.reshape(N, m, subdim)

        codes = np.empty((N, m), dtype=np.uint8)

        # Precompute codebook norms to speed up L2 distance
        # codebook_norms[j] shape (ksub,)
        codebook_norms = np.sum(pq_codebook * pq_codebook, axis=2)  # (m, ksub)

        # For each subspace compute distances and pick argmin
        for j in range(m):
            sub = subs[:, j, :]                            # (N, subdim)
            cb = pq_codebook[j]                           # (ksub, subdim)
            # dot = sub.dot(cb.T)                          # (N, ksub)
            dots = np.dot(sub, cb.T)                      # (N, ksub)
            x_norms = np.sum(sub * sub, axis=1, keepdims=True)  # (N,1)
            # dist = ||x||^2 + ||c||^2 - 2 * x.c
            dists = x_norms + codebook_norms[j].reshape(1, -1) - 2.0 * dots
            codes[:, j] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def load_inverted_list(self, cluster_id: int):
        # Load metadata
        meta_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        m = meta['pq']['m']
        posting_bytes = meta['pq']['posting_bytes']

        file_name = meta['cluster_files'][str(cluster_id)]
        file_path = os.path.join(self.index_path, file_name)

        if not os.path.exists(file_path):
            return np.empty(0, dtype=POSTING_DTYPE)

        file_size = os.path.getsize(file_path)
        count = file_size // posting_bytes

        arr = np.memmap(
            file_path,
            dtype=POSTING_DTYPE,
            mode='r',
            shape=(count,)
        )
        return arr

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        # ------------------------
        # 1️⃣ Normalize query
        # ------------------------
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q /= q_norm

        # ------------------------
        # 2️⃣ Load metadata & PQ codebook
        # ------------------------
        meta_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        n_clusters = int(meta["n_clusters"])
        num_records = int(meta["num_records"])
        m, ksub, d_sub = self.load_pq_codebook().shape  # PQ codebook shape
        codebook = self.load_pq_codebook()  # shape: (m, ksub, d_sub)

        # ------------------------
        # 3️⃣ Compute LUT (lookup table)
        # ------------------------
        q_subs = q.reshape(m, d_sub)
        LUT = np.empty((m, ksub), dtype=np.float32)
        for j in range(m):
            LUT[j] = np.sum((codebook[j] - q_subs[j]) ** 2, axis=1)

        # ------------------------
        # 4️⃣ Choose clusters to search (nprobe)
        # ------------------------
        centroids = self.load_centroids(n_clusters)
        sims_to_centroids = centroids.dot(q)
        del centroids  # free memory
        if num_records <= 1_000_000:
            nprobe = 3
        elif num_records <= 10_000_000:
            nprobe = 6
        elif num_records <= 20_000_000:
            nprobe = 8
        nprobe = min(max(1, nprobe), n_clusters)
        top_centroid_idxs = np.argpartition(-sims_to_centroids, nprobe-1)[:nprobe]
        del sims_to_centroids

        # ------------------------
        # 5️⃣ Load candidate IDs & PQ codes
        # ------------------------
        candidate_postings_list = []
        for ci in top_centroid_idxs:
            postings = self.load_inverted_list(int(ci))  # returns structured array with fields 'id' and 'code'
            if postings.size > 0:
                candidate_postings_list.append(np.array(postings))  # force copy to RAM
        del top_centroid_idxs
        if not candidate_postings_list:
            return []

        candidate_postings = np.concatenate(candidate_postings_list, axis=0)
        del candidate_postings_list
        n_candidates = candidate_postings.size

        # ------------------------
        # 6️⃣ Compute distances via ADC in batches
        # ------------------------
        target_batch_bytes = 8 * 1024 * 1024
        batch_size = max(1, target_batch_bytes // (m * ELEMENT_SIZE))
        batch_size = min(batch_size, n_candidates)
        all_distances = np.empty(n_candidates, dtype=np.float32)

        for batch_start in range(0, n_candidates, batch_size):
            batch_end = min(batch_start + batch_size, n_candidates)
            batch_postings = candidate_postings[batch_start:batch_end]

            # Compute distances using LUT
            # batch_postings['code'] shape: (batch_size, m)
            dists = np.zeros(batch_end - batch_start, dtype=np.float32)
            for j in range(m):
                dists += LUT[j, batch_postings['code'][:, j]]
            all_distances[batch_start:batch_end] = dists

        del candidate_postings
        del LUT

        # ------------------------
        # 7️⃣ Select top-k
        # ------------------------
        if top_k < n_candidates:
            topk_idx = np.argpartition(all_distances, top_k - 1)[:top_k]
            topk_idx = topk_idx[np.argsort(all_distances[topk_idx])]  # sort by distance
        else:
            topk_idx = np.argsort(all_distances)[:top_k]

        # ------------------------
        # 8️⃣ Return IDs
        # ------------------------
        result_ids = candidate_postings['id'][topk_idx]
        return [int(x) for x in result_ids]