import time
import redis
from pymongo import MongoClient
from cassandra.cluster import Cluster

MONGO_HOST, CASSANDRA_HOST, REDIS_HOST = (
    "penguin_mongo",
    "penguin_cassandra",
    "penguin_redis",
)


def run_benchmarks():
    mongo_coll = MongoClient(MONGO_HOST, 27017)["penguin_db"]["penguins"]
    cass_session = Cluster([CASSANDRA_HOST]).connect("penguin_ks")
    cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

    test_id = "P1020"
    iterations = 1000

    print(f"--- Starting Benchmarks ({iterations} iterations) ---")

    start = time.time()
    for _ in range(iterations):
        res = mongo_coll.find_one({"penguin_id": test_id})
    mongo_time = (time.time() - start) / iterations
    print(f"MongoDB Latency: {mongo_time * 1000:.4f} ms per read")

    query = cass_session.prepare(
        "SELECT * FROM penguins_by_island WHERE island='Biscoe' AND species='Adelie' AND penguin_id=?"
    )
    start = time.time()
    for _ in range(iterations):
        res = cass_session.execute(query, [test_id])
    cass_time = (time.time() - start) / iterations
    print(f"Cassandra Latency: {cass_time * 1000:.4f} ms per read")

    prediction_val = "0.0"
    cache.set(test_id, prediction_val)

    start = time.time()
    for _ in range(iterations):
        res = cache.get(test_id)
    redis_time = (time.time() - start) / iterations
    print(f"Redis Cache Latency: {redis_time * 1000:.4f} ms per read")

    print("\n--- Project Report: Comparative Performance ---")
    print("| Technology | Avg Latency | Throughput (est) |")
    print("|------------|-------------|------------------|")
    print(
        f"| MongoDB    | {mongo_time * 1000:.2f} ms   | {int(1 / mongo_time)} req/s      |"
    )
    print(
        f"| Cassandra  | {cass_time * 1000:.2f} ms   | {int(1 / cass_time)} req/s      |"
    )
    print(
        f"| Redis      | {redis_time * 1000:.2f} ms   | {int(1 / redis_time)} req/s      |"
    )


if __name__ == "__main__":
    run_benchmarks()
