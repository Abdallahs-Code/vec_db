from typing import Annotated
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import math

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        # Initialize a VecDB object.
        #
        # Args:
        #     database_file_path (str): The path to the file where the database will be stored.
        #     index_file_path (str): The path to the file where the index will be stored.
        #     new_db (bool): Whether a new database should be created.
        #     db_size (int): The number of records in the database if new_db is True.
        #
        # Raises:
        #     ValueError: If new_db is True and db_size is None.
        #
        # Returns:
        #     None
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        # Generate a random database of a given size.
        #
        # Args:
        #     size (int): The number of records in the database.
        #
        # Returns:
        #     None
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        # Write a numpy array of vectors to a file.
        #
        # Args:
        #     vectors (np.ndarray): The numpy array of vectors to write to the file.
        #
        # Returns:
        #     None
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        # Get the number of records in the database.
        #
        # Returns:
        #     int: The number of records in the database.
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        # Insert new records into the database.
        #
        # Args:
        #     rows (Annotated[np.ndarray, (int, 70)]): The numpy array of vectors to insert into the database.
        #
        # Returns:
        #     None
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        # NOTE: Bellow is the old code, it is not efficient. 
        # I enhanced the code beneath these couple of commented lines 
        # in order to make it more efficient as it is critical for the project  
        # now it checks for the index before starting the actual loading of the vector
        # try:
        #     offset = row_num * DIMENSION * ELEMENT_SIZE
        #     mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
        #     return np.array(mmap_vector[0])
        # except Exception as e:
        #     return f"An error occurred: {e}"
        num_records = self._get_num_records()
        if row_num < 0 or row_num >= num_records:
            raise ValueError(f"Invalid row number: {row_num}. Must be between 0 and {num_records - 1}.")
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

    # def get_all_rows(self) -> np.ndarray:
    #     # Get all the records from the database.
    #     #
    #     # Returns:
    #     #     np.ndarray: A numpy array of shape (num_records, DIMENSION) containing all the records in the database.
    #     #
    #     # Note:
    #     #     This function loads all the data in memory, so be careful when using it with large databases.
    #     num_records = self._get_num_records()
    #     vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
    #     return np.array(vectors)
    
    def _cal_score(self, vec1, vec2):
        # Calculate the cosine similarity between two vectors.
        #
        # Args:
        #     vec1 (np.ndarray): The first vector.
        #     vec2 (np.ndarray): The second vector.
        #
        # Returns:
        #     float: The cosine similarity between the two vectors.
        #
        # Note:
        #     The cosine similarity is calculated as the dot product of the two vectors divided by the product of their norms.
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def compute_clustering_parameters(self):
        # Compute parameters for clustering dynamically based on database size.
        # Sets self.sample_size, self.n_clusters, self.batch_size, self.max_iter.
        db_size = self._get_num_records()
        # Sample size: 5% of db, good enough for all db sizes since the lowest db size is 1M
        self.sample_size = int(0.05 * db_size)
        # Number of clusters: nearest power of 2 of sqrt(db_size) -> rule of thumb
        sqrt_db = math.sqrt(db_size)
        self.n_clusters = 2 ** int(round(math.log2(sqrt_db)))
        # MiniBatch batch size is 4096 which is good for db sizes 1M - 20M
        self.batch_size = 4096
        # Max iterations appropriate regardless of db size
        self.max_iter = 200

    def sample_for_kmeans(self, seed : int = DB_SEED_NUMBER) -> np.ndarray:
        # Return a numpy array of vectors randomly sampled from the database.
        #
        # Args:
        #     sample_size (int): The number of records to sample from the database.
        #     seed (int): The seed for the random number generator. Defaults to DB_SEED_NUMBER.
        #
        # Returns:
        #     np.ndarray: A numpy array of shape (sample_size, DIMENSION) containing the sampled vectors.
        #
        # Raises:
        #     ValueError: If the sample size is greater than the number of records in the database.
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
        # Train MiniBatchKMeans on sampled vectors and return centroids.
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=42
        )
        kmeans.fit(sampled_vectors)
        return kmeans.cluster_centers_.astype(np.float32)

    def save_centroids(self, centroids: np.ndarray):
        # Save centroids to disk as a raw binary file (no header, same format as main DB).
        #
        # Args:
        #     centroids (np.ndarray): Array of shape (n_clusters, 64), dtype=float32
        mmap_centroids = np.memmap(self.index_path, dtype=np.float32, mode='w+', shape=centroids.shape)
        mmap_centroids[:] = centroids[:]
        mmap_centroids.flush()

    def load_centroids(self, n_clusters: int) -> np.ndarray:
        # Load centroids from the raw binary file.
        #
        # Args:
        #     n_clusters (int): Number of clusters (needed to reshape the raw file)
        #
        # Returns:
        #     np.ndarray: Centroids of shape (n_clusters, 64), dtype=float32
        return np.memmap(self.index_path, dtype=np.float32, mode='r', shape=(n_clusters, DIMENSION))

    def _build_index(self):
        # Build the IVF index by training centroids on sampled vectors.
        self.compute_clustering_parameters()
        sampled_vectors = self.sample_for_kmeans()
        centroids = self.train_centroids(sampled_vectors)
        self.save_centroids(centroids)
        # (Actual assignment of vectors to centroids is handled later by inverted lists)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        # Retrieve the top-k most similar vectors from the database based on a given query vector.
        #
        # Args:
        #     query (Annotated[np.ndarray, (1, DIMENSION)]): The query vector.
        #     top_k (int): The number of most similar vectors to retrieve. Defaults to 5.
        #
        # Returns:
        #     List[int]: A list of the IDs of the top-k most similar vectors.
        #
        # Note:
        #     The similarity is calculated as the cosine similarity between the query vector and each vector in the database.
        #     If two vectors have the same score, the lowest ID is returned.
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]