import pandas as pd
import uuid
import requests
import io
from pymongo import MongoClient
from cassandra.cluster import Cluster

DATASET_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
MONGO_HOST = "penguin_mongo"
CASSANDRA_HOST = "penguin_cassandra"


def get_data():
    """Download and clean the dataset."""
    print(f"Downloading data from {DATASET_URL}...")
    s = requests.get(DATASET_URL).content
    df = pd.read_csv(io.StringIO(s.decode("utf-8")))

    initial_count = len(df)
    df = df.dropna()
    print(f"Data cleaned. Rows: {initial_count} -> {len(df)}")
    return df


def setup_mongodb(df):
    """Insert data into MongoDB with nested schema."""
    print(f"Connecting to MongoDB at {MONGO_HOST}...")
    client = MongoClient(MONGO_HOST, 27017)
    db = client["penguin_db"]
    collection = db["penguins"]

    collection.drop()

    documents = []
    for index, row in df.iterrows():
        p_id = f"P{1000 + index}"
        doc = {
            "penguin_id": p_id,
            "features": {
                "bill_length": row["bill_length_mm"],
                "bill_depth": row["bill_depth_mm"],
                "flipper_length": row["flipper_length_mm"],
                "body_mass": row["body_mass_g"],
            },
            "label": row["species"],
            "island": row["island"],
        }
        documents.append(doc)

    collection.insert_many(documents)
    print(f"Inserted {len(documents)} documents into MongoDB.")
    return df


def setup_cassandra(df):
    """Insert data into Cassandra with partitioned schema."""
    print(f"Connecting to Cassandra at {CASSANDRA_HOST}...")
    cluster = Cluster([CASSANDRA_HOST], port=9042)
    session = cluster.connect()

    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS penguin_ks 
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
    """)
    session.set_keyspace("penguin_ks")

    session.execute("""
        CREATE TABLE IF NOT EXISTS penguins_by_island (
            island TEXT,
            species TEXT,
            penguin_id UUID,
            bill_length FLOAT,
            body_mass INT,
            PRIMARY KEY ((island), species, penguin_id)
        )
    """)

    query = session.prepare("""
        INSERT INTO penguins_by_island (island, species, penguin_id, bill_length, body_mass)
        VALUES (?, ?, ?, ?, ?)
    """)

    count = 0
    for index, row in df.iterrows():
        u_id = uuid.uuid4()
        session.execute(
            query,
            (
                row["island"],
                row["species"],
                u_id,
                float(row["bill_length_mm"]),
                int(row["body_mass_g"]),
            ),
        )
        count += 1

    print(f"Inserted {count} rows into Cassandra.")
    cluster.shutdown()


if __name__ == "__main__":
    df = get_data()
    setup_mongodb(df)
    setup_cassandra(df)
    print("\n--- SUCCESS: Data Ingestion Complete ---")
