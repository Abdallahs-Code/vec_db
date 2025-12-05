from typing import Annotated
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import math
import json
import time
import gc
from pathlib import Path

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
DB_SEED_NUMBER = 42

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index", 
                 new_db=True, db_size=None, use_pq=True) -> None:
        """
        VecDB with optional Product Quantization for extreme compression.
        
        Parameters:
        -----------
        use_pq : bool
            If True, use Product Quantization to compress vectors (default: True)
        """
        if not os.path.isfile(database_file_path):
            print("Database file does NOT exist.")
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.use_pq = use_pq
        
        # Product Quantization parameters
        if use_pq:
            self.M = 8  # Number of sub-vectors (64/8 = 8D per sub-vector)
            self.K = 256  # Number of centroids per sub-quantizer (fits in uint8)
            self.d_sub = DIMENSION // self.M  # Dimension of each sub-vector
        
        self._build_index()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]) -> None:
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        new_total_records = num_old_records + num_new_records
        with open(self.db_path, 'ab') as f:
            f.truncate(new_total_records * DIMENSION * ELEMENT_SIZE)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', 
                                 shape=(new_total_records, DIMENSION))
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        num_records = self._get_num_records()
        if row_num < 0 or row_num >= num_records:
            raise ValueError(f"Invalid row number: {row_num}")
        offset = row_num * DIMENSION * ELEMENT_SIZE
        mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                                shape=(1, DIMENSION), offset=offset)
        return np.array(mmap_vector[0])

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                           shape=(num_records, DIMENSION))
        return np.array(vectors)

    def compute_clustering_parameters(self):
        db_size = self._get_num_records()
        self.n_clusters = 1024  # Fixed IVF clusters
        self.sample_size = min(int(0.03 * db_size), 100000)
        self.batch_size = 4096
        self.max_iter = 200

    def train_product_quantizer(self, sample_vectors: np.ndarray) -> dict:
        """
        Train Product Quantization codebooks.
        
        Returns:
        --------
        dict with keys:
            'codebooks': list of M codebooks, each (K, d_sub)
        """
        print(f"Training PQ with M={self.M} sub-quantizers, K={self.K} centroids each...")
        
        codebooks = []
        
        for m in range(self.M):
            # Extract m-th sub-vector from all samples
            start_dim = m * self.d_sub
            end_dim = (m + 1) * self.d_sub
            sub_vectors = sample_vectors[:, start_dim:end_dim]
            
            # Train k-means on this sub-space
            kmeans = MiniBatchKMeans(
                n_clusters=self.K,
                batch_size=min(4096, len(sub_vectors)),
                max_iter=100,
                random_state=42 + m
            )
            kmeans.fit(sub_vectors)
            codebooks.append(kmeans.cluster_centers_.astype(np.float32))
            
            print(f"  Trained sub-quantizer {m+1}/{self.M}")
        
        return {'codebooks': codebooks}

    def encode_vector_pq(self, vector: np.ndarray, codebooks: list) -> np.ndarray:
        """
        Encode a single vector using Product Quantization.
        
        Returns:
        --------
        codes : np.ndarray of shape (M,) with dtype uint8
            The PQ codes for the vector
        """
        codes = np.zeros(self.M, dtype=np.uint8)
        
        for m in range(self.M):
            start_dim = m * self.d_sub
            end_dim = (m + 1) * self.d_sub
            sub_vector = vector[start_dim:end_dim]
            
            # Find nearest centroid in this sub-space
            distances = np.sum((codebooks[m] - sub_vector) ** 2, axis=1)
            codes[m] = np.argmin(distances)
        
        return codes

    def encode_vectors_pq_batch(self, vectors: np.ndarray, codebooks: list) -> np.ndarray:
        """
        Encode multiple vectors using PQ (batched for efficiency).
        
        Returns:
        --------
        codes : np.ndarray of shape (n_vectors, M) with dtype uint8
        """
        n_vectors = len(vectors)
        codes = np.zeros((n_vectors, self.M), dtype=np.uint8)
        
        for m in range(self.M):
            start_dim = m * self.d_sub
            end_dim = (m + 1) * self.d_sub
            sub_vectors = vectors[:, start_dim:end_dim]
            
            # Compute distances to all centroids for this sub-quantizer
            # Shape: (n_vectors, K)
            distances = np.sum((sub_vectors[:, np.newaxis, :] - codebooks[m][np.newaxis, :, :]) ** 2, axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        
        return codes

    def compute_pq_distance_tables(self, query: np.ndarray, codebooks: list) -> np.ndarray:
        """
        Pre-compute distance tables for asymmetric PQ search.
        
        Returns:
        --------
        distance_tables : np.ndarray of shape (M, K)
            distance_tables[m, k] = ||query_subvec_m - codebook_m[k]||^2
        """
        distance_tables = np.zeros((self.M, self.K), dtype=np.float32)
        
        for m in range(self.M):
            start_dim = m * self.d_sub
            end_dim = (m + 1) * self.d_sub
            query_sub = query[start_dim:end_dim]
            
            # Compute squared distances to all centroids in this sub-space
            distance_tables[m] = np.sum((codebooks[m] - query_sub) ** 2, axis=1)
        
        return distance_tables

    def estimate_distance_from_pq(self, codes: np.ndarray, distance_tables: np.ndarray) -> float:
        """
        Estimate squared L2 distance using PQ codes and pre-computed tables.
        
        Parameters:
        -----------
        codes : np.ndarray of shape (M,) with dtype uint8
        distance_tables : np.ndarray of shape (M, K)
        
        Returns:
        --------
        estimated_distance : float
        """
        distance = 0.0
        for m in range(self.M):
            distance += distance_tables[m, codes[m]]
        return distance

    def save_pq_codebooks(self, codebooks: list):
        """Save PQ codebooks to disk."""
        codebooks_path = os.path.join(self.index_path, "pq_codebooks.dat")
        
        # Stack all codebooks: shape (M, K, d_sub)
        codebooks_array = np.stack(codebooks, axis=0).astype(np.float32)
        
        with open(codebooks_path, 'wb') as f:
            codebooks_array.tofile(f)
        
        # Save metadata
        pq_meta = {
            'M': self.M,
            'K': self.K,
            'd_sub': self.d_sub
        }
        with open(os.path.join(self.index_path, "pq_meta.json"), 'w') as f:
            json.dump(pq_meta, f)

    def load_pq_codebooks(self) -> list:
        """Load PQ codebooks from disk."""
        codebooks_path = os.path.join(self.index_path, "pq_codebooks.dat")
        
        if not os.path.exists(codebooks_path):
            raise FileNotFoundError("PQ codebooks not found")
        
        # Load metadata
        with open(os.path.join(self.index_path, "pq_meta.json"), 'r') as f:
            pq_meta = json.load(f)
        
        M = pq_meta['M']
        K = pq_meta['K']
        d_sub = pq_meta['d_sub']
        
        # Load codebooks
        with open(codebooks_path, 'rb') as f:
            codebooks_array = np.fromfile(f, dtype=np.float32, count=M * K * d_sub)
        
        codebooks_array = codebooks_array.reshape(M, K, d_sub)
        
        return [codebooks_array[m] for m in range(M)]

    def save_pq_codes(self, codes: np.ndarray, cluster_id: int):
        """Save PQ codes for a cluster."""
        codes_path = os.path.join(self.index_path, f"cluster_{cluster_id}_pq.dat")
        with open(codes_path, 'wb') as f:
            codes.tofile(f)

    def load_pq_codes(self, cluster_id: int, count: int) -> np.ndarray:
        """Load PQ codes for a cluster."""
        codes_path = os.path.join(self.index_path, f"cluster_{cluster_id}_pq.dat")
        
        if not os.path.exists(codes_path) or os.path.getsize(codes_path) == 0:
            return np.array([], dtype=np.uint8).reshape(0, self.M)
        
        with open(codes_path, 'rb') as f:
            codes = np.fromfile(f, dtype=np.uint8, count=count * self.M)
        
        return codes.reshape(count, self.M)

    def sample_for_kmeans(self, seed: int = DB_SEED_NUMBER) -> np.ndarray:
        sample_size = self.sample_size
        num_records = self._get_num_records()
        if sample_size > num_records:
            raise ValueError("Sample size cannot be greater than number of records")
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(num_records, sample_size, replace=False)
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                              shape=(num_records, DIMENSION))
        samples = db_vectors[sample_indices]
        return np.array(samples, dtype=np.float32)

    def train_centroids(self, sampled_vectors: np.ndarray) -> np.ndarray:
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=42
        )
        kmeans.fit(sampled_vectors)
        return kmeans.cluster_centers_.astype(np.float32)
    
    def save_centroids(self, centroids: np.ndarray):
        if centroids.dtype != np.float32:
            centroids = centroids.astype(np.float32)
        clusters_file_path = os.path.join(self.index_path, "centroids.dat")
        with open(clusters_file_path, 'wb') as f:
            centroids.tofile(f)

    def load_centroids(self, n_clusters: int) -> np.ndarray:
        clusters_file_path = os.path.join(self.index_path, "centroids.dat")
        if not os.path.exists(clusters_file_path):
            raise FileNotFoundError("Centroids file not found")
        with open(clusters_file_path, 'rb') as f:
            centroids = np.fromfile(f, dtype=np.float32, count=n_clusters * DIMENSION)
        return centroids.reshape(n_clusters, DIMENSION)

    def _build_index(self) -> None:
        self.compute_clustering_parameters()
        need_rebuild = self._should_rebuild_index()
        if need_rebuild:
            print("Building new index...")
            self._cleanup_old_index()
        else:
            print("Index is up-to-date. No rebuild needed.")
            return
        
        os.makedirs(self.index_path, exist_ok=True)
        
        # Sample vectors for training
        sampled_vectors = self.sample_for_kmeans()
        
        # Train IVF centroids
        centroids = self.train_centroids(sampled_vectors)
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroid_norms[centroid_norms == 0] = 1.0
        centroids = (centroids / centroid_norms).astype(np.float32)
        self.save_centroids(centroids)
        
        # Train PQ codebooks if enabled
        if self.use_pq:
            pq_data = self.train_product_quantizer(sampled_vectors)
            self.save_pq_codebooks(pq_data['codebooks'])
            codebooks = pq_data['codebooks']
        
        del sampled_vectors
        gc.collect()
        
        num_records = self._get_num_records()
        chunk_size = 65536
        
        # Create cluster files
        cluster_paths = []
        for ci in range(self.n_clusters):
            cluster_file = os.path.join(self.index_path, f"cluster_{ci}.ids")
            with open(cluster_file, 'wb') as f:
                pass
            cluster_paths.append(cluster_file)
        
        cluster_counts = [0] * self.n_clusters
        
        # If using PQ, store codes per cluster
        if self.use_pq:
            cluster_pq_codes = [[] for _ in range(self.n_clusters)]
        
        # Assign vectors to clusters
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', 
                              shape=(num_records, DIMENSION))
        
        for chunk_start in range(0, num_records, chunk_size):
            chunk_end = min(num_records, chunk_start + chunk_size)
            chunk_vectors = np.array(db_vectors[chunk_start:chunk_end], dtype=np.float32)
            
            # Normalize and assign to clusters
            chunk_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
            chunk_norms[chunk_norms == 0] = 1.0
            chunk_vectors_normalized = chunk_vectors / chunk_norms
            similarities = np.dot(chunk_vectors_normalized, centroids.T)
            cluster_assignments = np.argmax(similarities, axis=1)
            
            # Encode with PQ if enabled
            if self.use_pq:
                chunk_codes = self.encode_vectors_pq_batch(chunk_vectors, codebooks)
            
            for cluster_id in range(self.n_clusters):
                mask = (cluster_assignments == cluster_id)
                local_indices = np.where(mask)[0]
                if local_indices.size == 0:
                    continue
                
                global_ids = (chunk_start + local_indices).astype(np.uint32)
                
                # Save IDs
                with open(cluster_paths[cluster_id], 'ab') as f:
                    global_ids.tofile(f)
                
                # Save PQ codes if enabled
                if self.use_pq:
                    cluster_pq_codes[cluster_id].append(chunk_codes[local_indices])
                
                cluster_counts[cluster_id] += len(global_ids)
            
            del chunk_vectors
            del chunk_vectors_normalized
            if self.use_pq:
                del chunk_codes
            gc.collect()
        
        del db_vectors
        
        # Save PQ codes per cluster
        if self.use_pq:
            for cluster_id in range(self.n_clusters):
                if cluster_pq_codes[cluster_id]:
                    all_codes = np.vstack(cluster_pq_codes[cluster_id])
                    self.save_pq_codes(all_codes, cluster_id)
            del cluster_pq_codes
        
        # Save metadata
        metadata = {
            "n_clusters": int(self.n_clusters),
            "num_records": int(num_records),
            "dimension": int(DIMENSION),
            "use_pq": self.use_pq,
            "cluster_files": {str(i): os.path.basename(cluster_paths[i]) 
                            for i in range(self.n_clusters)},
            "counts": {str(i): int(cluster_counts[i]) 
                      for i in range(self.n_clusters)},
            "build_timestamp": time.time()
        }
        
        with open(os.path.join(self.index_path, "index_meta.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Index build complete.")

    def _should_rebuild_index(self) -> bool:
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        if not os.path.exists(centroids_path):
            return True
        metadata_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(metadata_path):
            return True
        try:
            with open(metadata_path, 'r') as f:
                old_metadata = json.load(f)
            if old_metadata.get("n_clusters") != self.n_clusters:
                return True
            if old_metadata.get("num_records") != self._get_num_records():
                return True
            if old_metadata.get("use_pq") != self.use_pq:
                return True
            return False
        except:
            return True

    def _cleanup_old_index(self):
        if os.path.isdir(self.index_path):
            import shutil
            shutil.rmtree(self.index_path)

    def load_inverted_list(self, cluster_id: int) -> np.ndarray:
        meta_path = os.path.join(self.index_path, "index_meta.json")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        file_name = meta["cluster_files"].get(str(cluster_id))
        if file_name is None:
            return np.array([], dtype=np.uint32)
        file_path = os.path.join(self.index_path, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return np.array([], dtype=np.uint32)
        count = os.path.getsize(file_path) // np.dtype(np.uint32).itemsize
        with open(file_path, 'rb') as f:
            return np.fromfile(f, dtype=np.uint32, count=count)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieve with optional PQ acceleration.
        """
        qn = np.asarray(query, dtype=np.float32).reshape(-1)
        if qn.size != DIMENSION:
            raise ValueError(f"Query dimension mismatch")
        q_norm = np.linalg.norm(qn)
        if q_norm > 0:
            qn /= q_norm
        
        meta_path = os.path.join(self.index_path, "index_meta.json")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        n_clusters = int(meta["n_clusters"])
        num_records = int(meta["num_records"])
        use_pq = meta.get("use_pq", False)
        
        centroids = self.load_centroids(n_clusters)
        sims_to_centroids = centroids.dot(qn)
        del centroids
        
        if num_records <= 1_000_000:
            nprobe = 2
        elif num_records <= 10_000_000:
            nprobe = 3
        else:
            nprobe = 4
        nprobe = min(max(1, nprobe), n_clusters)
        
        top_centroid_idxs = np.argpartition(-sims_to_centroids, nprobe-1)[:nprobe]
        del sims_to_centroids
        
        # Load PQ codebooks if using PQ
        if use_pq:
            codebooks = self.load_pq_codebooks()
            distance_tables = self.compute_pq_distance_tables(qn, codebooks)
        
        # Gather candidates
        import heapq
        top_heap = []
        
        for ci in top_centroid_idxs:
            ids = self.load_inverted_list(int(ci))
            if ids.size == 0:
                continue
            
            if use_pq:
                # Use PQ for fast filtering
                codes = self.load_pq_codes(int(ci), len(ids))
                
                # Estimate distances using PQ
                for idx_pos, vec_id in enumerate(ids):
                    pq_dist = self.estimate_distance_from_pq(codes[idx_pos], distance_tables)
                    # Convert distance to similarity (negative distance)
                    score = -pq_dist
                    
                    if len(top_heap) < top_k * 2:  # Keep 2x for reranking
                        heapq.heappush(top_heap, (score, int(vec_id)))
                    elif score > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (score, int(vec_id)))
                
                del codes
            else:
                # Original non-PQ method
                db_file = open(self.db_path, 'rb')
                for vec_id in ids:
                    offset = int(vec_id) * DIMENSION * ELEMENT_SIZE
                    db_file.seek(offset)
                    vec_bytes = db_file.read(DIMENSION * ELEMENT_SIZE)
                    vec = np.frombuffer(vec_bytes, dtype=np.float32)
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm > 0:
                        vec = vec / vec_norm
                    score = float(np.dot(vec, qn))
                    
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, int(vec_id)))
                    elif score > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (score, int(vec_id)))
                db_file.close()
            
            del ids
        
        # If using PQ, rerank top candidates with exact distances
        if use_pq and len(top_heap) > top_k:
            # Rerank top 2*top_k candidates
            candidates = sorted(top_heap, reverse=True)[:top_k * 2]
            
            refined_heap = []
            db_file = open(self.db_path, 'rb')
            for _, vec_id in candidates:
                offset = int(vec_id) * DIMENSION * ELEMENT_SIZE
                db_file.seek(offset)
                vec_bytes = db_file.read(DIMENSION * ELEMENT_SIZE)
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 0:
                    vec = vec / vec_norm
                exact_score = float(np.dot(vec, qn))
                refined_heap.append((exact_score, vec_id))
            db_file.close()
            
            refined_heap.sort(key=lambda x: (-x[0], x[1]))
            result = [x[1] for x in refined_heap[:top_k]]
        else:
            top_heap.sort(key=lambda x: (-x[0], x[1]))
            result = [x[1] for x in top_heap[:top_k]]
        
        gc.collect()
        return result
