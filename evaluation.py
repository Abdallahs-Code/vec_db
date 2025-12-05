import os
import random
import time
import gc
import tracemalloc
import numpy as np
from dataclasses import dataclass
from typing import List

try:
    from memory_profiler import memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("Warning: memory_profiler not installed. Memory usage will not be measured.")

TEAM_NUMBER = 1
SEED_NUMBER = 10
random.seed(SEED_NUMBER)

PATH_DB_1M = "idx_1M"
PATH_DB_10M = "idx_10M"
PATH_DB_20M = "idx_20M"

PATH_DB_VECTORS_1M = "OpenSubtitles_en_1M_emb_64.dat"
PATH_DB_VECTORS_10M = "OpenSubtitles_en_10M_emb_64.dat"
PATH_DB_VECTORS_20M = "OpenSubtitles_en_20M_emb_64.dat"

DIMENSION = 64
queries_embed_file = "queries_emb_64.dat"
actual_sorted_ids_file = "actual_sorted_ids_20m.dat"
saved_top_k = 30_000
needed_top_k = 10_000

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, queries, top_k, actual_ids, num_runs):
    results = []
    for i in range(num_runs):
        tic = time.time()
        db_ids = db.retrieve(queries[i], top_k)
        toc = time.time()
        results.append(Result(toc - tic, top_k, db_ids, actual_ids[i]))
    return results

def memory_usage_run_queries(db, queries, top_k, actual_ids, num_runs):
    """Run queries and measure memory safely on Windows."""
    if not MEMORY_PROFILER_AVAILABLE:
        return run_queries(db, queries, top_k, actual_ids, num_runs), -1

    mem_before = max(memory_usage())
    results = run_queries(db, queries, top_k, actual_ids, num_runs)
    mem_after = max(memory_usage())
    return results, mem_after - mem_before

def evaluate_result(results: List[Result]):
    if len(results) == 0:
        return 0, 0
    scores, run_time = [], []
    for res in results:
        run_time.append(res.run_time)
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append(-1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)
    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_actual_ids_first_k(actual_sorted_ids, k, out_len=10_000):
    return [[id for id in actual_sorted_ids_one_q if id < k] for actual_sorted_ids_one_q in actual_sorted_ids][:out_len]

def prepare_queries():
    if not os.path.exists(queries_embed_file):
        from sentence_transformers import SentenceTransformer
        batch_sentences = [
            "Hello World",
            "We are Software Engineering Students",
            "What's the best way to be a good human?",
            "What a good day"
        ]
        model = SentenceTransformer('minishlab/potion-base-2M')
        queries_np = model.encode(batch_sentences, convert_to_numpy=True).astype(np.float32)
        queries_np.tofile(queries_embed_file)
    else:
        queries_np = np.fromfile(queries_embed_file, dtype=np.float32).reshape(-1, DIMENSION)
    query_dummy = queries_np[0].reshape(1, DIMENSION)
    queries = [queries_np[1].reshape(1, DIMENSION),
               queries_np[2].reshape(1, DIMENSION),
               queries_np[3].reshape(1, DIMENSION)]
    queries_np = queries_np[1:, :]
    return query_dummy, queries, queries_np

def prepare_actual_ids(queries_np):
    if not os.path.exists(actual_sorted_ids_file):
        vectors = np.memmap(PATH_DB_VECTORS_20M, dtype='float32', mode='r', shape=(20_000_000, DIMENSION))
        actual_sorted_ids_20m = np.argsort(
            np.dot(vectors, queries_np.T) /
            (1e-45 + np.linalg.norm(vectors, axis=1)[:, None] * np.linalg.norm(queries_np, axis=1)),
            axis=0
        )[-saved_top_k:][::-1].T.astype(np.int32)
        actual_sorted_ids_20m.tofile(actual_sorted_ids_file)
    else:
        actual_sorted_ids_20m = np.fromfile(actual_sorted_ids_file, dtype=np.int32).reshape(-1, saved_top_k)
    return actual_sorted_ids_20m

def main():
    global queries_np

    query_dummy, queries, queries_np = prepare_queries()
    actual_sorted_ids_20m = prepare_actual_ids(queries_np)

    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    from vec_db import VecDB
    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    # for stat in stats[:5]:
    #     print(stat)
    tracemalloc.stop()

    print("Team Number", TEAM_NUMBER)

    database_info = {
        "1M": {"database_file_path": PATH_DB_VECTORS_1M, "index_file_path": PATH_DB_1M, "size": 10**6},
        "10M": {"database_file_path": PATH_DB_VECTORS_10M, "index_file_path": PATH_DB_10M, "size": 10 * 10**6},
        "20M": {"database_file_path": PATH_DB_VECTORS_20M, "index_file_path": PATH_DB_20M, "size": 20 * 10**6}
    }

    for db_name, info in database_info.items():
        print("*" * 55)
        print(f"Evaluating DB of size {db_name}")

        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        db = VecDB(database_file_path=info["database_file_path"], index_file_path=info["index_file_path"], new_db=False)
        end_snapshot = tracemalloc.take_snapshot()
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        # for stat in stats[:5]:
        #     print(stat)
        tracemalloc.stop()

        actual_ids = get_actual_ids_first_k(actual_sorted_ids_20m, info["size"], needed_top_k)

        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        _ = run_queries(db, query_dummy, 5, actual_ids, 1)
        end_snapshot = tracemalloc.take_snapshot()
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        # for stat in stats[:5]:
        #     print(stat)
        tracemalloc.stop()

        res, mem = memory_usage_run_queries(db, queries, 5, actual_ids, 3)
        eval_score = evaluate_result(res)
        to_print = f"{db_name}\tscore\t{eval_score[0]}\ttime\t{eval_score[1]:.2f}\tRAM\t{mem:.2f} MB"
        print(to_print)

        del db, actual_ids, res, mem, eval_score
        gc.collect()

if __name__ == "__main__":
    main()