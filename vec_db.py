from typing import Annotated
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import math
import json
import time
from heapq import heappushpop, heappush
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
        # Sample size: 10% of db, good enough for all db sizes since the lowest db size is 1M
        self.sample_size = int(0.1 * db_size)
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
        # Train PQ codebook using MiniBatchKMeans
        m = PQ_M
        ksub = PQ_KSUB
        d = DIMENSION // m
        assert samples.ndim == 2 and samples.shape[1] == DIMENSION, \
            f"Expected samples of shape (N, {DIMENSION}), got {samples.shape}"
        # Reshape into subvectors
        residual_subvectors = samples.reshape(-1, m, d)
        codebooks = np.zeros((m, ksub, d), dtype=np.float32)
        # Train subquantizers independently
        for si in range(m):
            sub_data = residual_subvectors[:, si, :]
            from sklearn.cluster import MiniBatchKMeans
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
        # Validate codebook shape
        expected_shape = (PQ_M, PQ_KSUB, DIMENSION // PQ_M)
        if codebooks.shape != expected_shape:
            raise ValueError(f"Codebook shape mismatch: expected {expected_shape}, got {codebooks.shape}")
        file_path = os.path.join(self.index_path, "pq_codebook.dat")
        with open(file_path, "wb") as f:
            codebooks.astype(np.float32).tofile(f)
        # Validate file size
        expected_size = PQ_M * PQ_KSUB * (DIMENSION // PQ_M) * ELEMENT_SIZE
        actual_size = os.path.getsize(file_path)
        if expected_size != actual_size:
            raise RuntimeError(
                f"Codebook file corrupted: expected={expected_size}, actual={actual_size}"
            )
        
    def load_pq_codebook(self) -> np.ndarray:
        # Load PQ codebook from disk
        file_path = os.path.join(self.index_path, "pq_codebook.dat")
        if not os.path.exists(file_path):
            raise FileNotFoundError("PQ codebook not found — build index first")
        # Validate file size
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
            return
        # Create index directory
        os.makedirs(self.index_path, exist_ok=True)
        # Sample vectors for training
        sampled_vectors = self.sample_for_kmeans()
        # Get or create centroids
        centroids = self._get_or_create_centroids(sampled_vectors)
        # Get or create PQ codebook
        pq_codebook = self._get_or_create_pq_codebook(sampled_vectors)
        # Prepare for vector assignment
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
            # Process in chunks
            for chunk_start in range(0, num_records, chunk_size):
                chunk_end = min(num_records, chunk_start + chunk_size)
                # Load chunk into memory (raw vectors)
                chunk_vectors = np.array(db_vectors[chunk_start:chunk_end], dtype=np.float32)
                # Normalize vectors for centroid assignment
                chunk_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
                chunk_norms[chunk_norms == 0] = 1.0
                chunk_vectors_normalized = chunk_vectors / chunk_norms
                # Compute similarities to centroids
                similarities = np.dot(chunk_vectors_normalized, centroids.T)
                cluster_assignments = np.argmax(similarities, axis=1)
                # Encode vectors into PQ codes
                chunk_codes = self._encode_chunk_to_pq(chunk_vectors, pq_codebook)
                # Prepare posting dtype
                posting_dtype = np.dtype([('id', np.uint32), ('code', np.uint8, (PQ_M,))])
                # Write vector IDs + codes to cluster files
                for cluster_id in range(self.n_clusters):
                    mask = (cluster_assignments == cluster_id)
                    local_indices = np.where(mask)[0]
                    if local_indices.size == 0:
                        continue
                    # Global IDs
                    global_ids = (chunk_start + local_indices).astype(np.uint32)
                    # Codes to write
                    codes_to_write = chunk_codes[local_indices]
                    # Build structured array
                    arr = np.empty(local_indices.size, dtype=posting_dtype)
                    arr['id'] = global_ids
                    arr['code'] = codes_to_write
                    # Append to cluster file
                    with open(cluster_paths[cluster_id], 'ab') as f:
                        arr.tofile(f)
                    cluster_counts[cluster_id] += local_indices.size
                # Free memory
                del chunk_vectors, chunk_vectors_normalized, similarities
                del cluster_assignments, chunk_codes
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
                "posting_bytes": (4 + PQ_M),
                "trained_on": "raw_vectors"  
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
        metadata_path = os.path.join(self.index_path, "index_meta.json")
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        pq_path = os.path.join(self.index_path, "pq_codebook.dat")
        # If any essential file is missing, rebuild
        if not os.path.exists(metadata_path):
            return True
        if not os.path.exists(centroids_path):
            return True
        if not os.path.exists(pq_path):
            return True
        try:
            # Load metadata
            meta = json.load(open(metadata_path, "r"))
            # Validate number of clusters
            old_n_clusters = int(meta.get("n_clusters", 0))
            if old_n_clusters != self.n_clusters:
                print(f"n_clusters mismatch: old={old_n_clusters}, new={self.n_clusters}")
                return True
            # Validate number of records
            old_num_records = int(meta.get("num_records", 0))
            current_num_records = self._get_num_records()
            if old_num_records != current_num_records:
                print(f"num_records mismatch: old={old_num_records}, new={current_num_records}")
                return True
            # Validate centroids file size
            expected_centroid_size = old_n_clusters * DIMENSION * ELEMENT_SIZE
            actual_centroid_size = os.path.getsize(centroids_path)
            if expected_centroid_size != actual_centroid_size:
                print(f"Centroid file corrupted: expected={expected_centroid_size}, actual={actual_centroid_size}")
                return True
            # Validate PQ codebook size
            pq_meta = meta.get("pq", {})
            expected_pq_size = pq_meta.get("m", 0) * pq_meta.get("ksub", 0) * pq_meta.get("subdim", 0) * ELEMENT_SIZE
            actual_pq_size = os.path.getsize(pq_path)
            if expected_pq_size != actual_pq_size:
                print(f"PQ codebook corrupted: expected={expected_pq_size}, actual={actual_pq_size}")
                return True
            # Validate all cluster files exist
            cluster_files = meta.get("cluster_files", {})
            for cluster_id, fname in cluster_files.items():
                cluster_path = os.path.join(self.index_path, fname)
                if not os.path.exists(cluster_path):
                    print(f"Cluster file missing: {cluster_path}")
                    return True
            # All checks passed
            return False
        except Exception as e:
            print(f"Error reading index metadata or files: {e}")
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
            # Train k-means
            centroids = self.train_centroids(sampled_vectors)
            # Normalize centroids
            centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroid_norms[centroid_norms == 0] = 1.0
            centroids = (centroids / centroid_norms).astype(np.float32)
            # Save to disk
            self.save_centroids(centroids)
            print(f"Centroids trained and saved...")
            return centroids
        else:
            # Load existing centroids
            print(f"Loading existing centroids from {self.index_path}")
            centroids = self.load_centroids(self.n_clusters)
            return centroids
        
    def _get_or_create_pq_codebook(self, samples: np.ndarray) -> np.ndarray:
        pq_file = os.path.join(self.index_path, "pq_codebook.dat")
        # Load if exists
        if os.path.exists(pq_file):
            return self.load_pq_codebook()
        # Otherwise train & save
        codebooks = self.train_pq_codebook(samples)
        self.save_pq_codebook(codebooks)
        print(f"PQ codebook trained and saved...")
        return codebooks    
    
    def _encode_chunk_to_pq(self, chunk_vectors: np.ndarray, pq_codebook: np.ndarray) -> np.ndarray:
        # Encode a chunk of vectors into PQ codes
        if chunk_vectors.size == 0:
            return np.zeros((0, PQ_M), dtype=np.uint8)
        m = PQ_M
        subdim = DIMENSION // m
        ksub = PQ_KSUB
        assert pq_codebook.shape == (m, ksub, subdim), \
            f"pq_codebook shape mismatch: {pq_codebook.shape}"
        N = chunk_vectors.shape[0]
        subs = chunk_vectors.reshape(N, m, subdim)
        codes = np.empty((N, m), dtype=np.uint8)
        for j in range(m):
            sub = subs[:, j, :]             
            cb = pq_codebook[j]              
            # Compute squared Euclidean distances
            dots = np.dot(sub, cb.T)       
            sub_sq = np.sum(sub * sub, axis=1, keepdims=True)  
            cent_sq = np.sum(cb * cb, axis=1)                  
            # Distance
            dists = sub_sq - 2.0 * dots + cent_sq[None, :]     
            # Select centroid with mimimum distance
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
        # Load cluster file
        file_name = meta['cluster_files'][str(cluster_id)]
        file_path = os.path.join(self.index_path, file_name)
        # If file does not exist, return empty array
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
        # Prepare query
        q = np.asarray(query, dtype=np.float32).reshape(DIMENSION)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0
        q_normalized = q / q_norm
        # Load metadata & resources
        meta_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        n_clusters = int(meta["n_clusters"])
        num_records = int(meta["num_records"])
        # Load centroids and PQ codebook
        centroids = self.load_centroids(n_clusters)
        codebook = self.load_pq_codebook()
        m, ksub, d_sub = codebook.shape
        # Set candidates size based on DB size
        if num_records <= 1_000_000:
            candidates = 100
        elif num_records <= 10_000_000:
            candidates = 500
        else:
            candidates = 1000
        if candidates < top_k:
            candidates = top_k
        # Set nprobe
        nprobe = min(66, n_clusters)
        # Select top nprobe clusters using cosine similarity
        sims_to_centroids = centroids @ q_normalized
        nearest_centroids = np.argsort(sims_to_centroids)[::-1][:nprobe]
        # Build lookup table for asymmetric distance computation
        # LUT[m, k] = distance between query subvector m and codebook centroid k
        lut = np.empty((m, ksub), dtype=np.float32)
        for j in range(m):
            q_sub = q[j * d_sub:(j + 1) * d_sub]
            q_sq = np.sum(q_sub * q_sub)
            cent = codebook[j]
            dots = cent @ q_sub 
            cent_sq = np.sum(cent * cent, axis=1)
            # Squared Euclidean distance
            lut[j] = q_sq - 2.0 * dots + cent_sq
        # Collect candidates from selected clusters using a min-heap
        heap = []
        for cid in nearest_centroids:
            # Load inverted list for this cluster
            postings = self.load_inverted_list(int(cid))
            if postings.size == 0:
                continue
            # Compute approximate distances using LUT
            codes = postings['code']
            ids = postings['id']      
            # For each vector sum distances across all subquantizers
            approx_dists = np.sum(lut[np.arange(m)[:, None], codes.T], axis=0)
            # Maintain top candidates
            for vid, dist in zip(ids, approx_dists):
                neg_dist = -dist  # Negate for max-heap behavior
                if len(heap) < candidates:
                    heappush(heap, (neg_dist, int(vid)))
                else:
                    if neg_dist > heap[0][0]:  # If better than worst in heap
                        heappushpop(heap, (neg_dist, int(vid)))
        # Check if we found any candidates
        if not heap:
            return []
        # Extract candidate IDs (sort by distance for consistent ordering)
        heap_items = sorted([(-d, vid) for d, vid in heap], key=lambda x: x[0])
        candidates_ids = [vid for _, vid in heap_items]
        # Reorder using full vectors with cosine similarity
        scores = []
        for vid in candidates_ids:
            vec = self.get_one_row(vid)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                vec_norm = 1.0
            # Cosine similarity
            cos_sim = np.dot(q, vec) / (q_norm * vec_norm)
            scores.append((cos_sim, vid))
        # Sort by cosine similarity
        scores.sort(key=lambda x: x[0], reverse=True)
        # Return top_k IDs
        top_ids = [int(vid) for _, vid in scores[:top_k]]
        return top_ids