import pandas as pd
import os
from pymongo import MongoClient
from cassandra.cluster import Cluster

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
CUSTOM_DATA_PATH = "/home/jovyan/work/data/custom_upload.csv"

MONGO_URI = "mongodb://penguin_mongo:27017/"
CASSANDRA_HOST = "penguin_cassandra"


def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client["penguin_db"]
    return db["penguins"]


def get_cassandra_session():
    cluster = Cluster([CASSANDRA_HOST])
    session = cluster.connect()
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS penguin_ks 
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
    """)
    session.set_keyspace("penguin_ks")
    return session


def ingest_data():
    print("--- Starting Data Ingestion Pipeline ---")

    if os.path.exists(CUSTOM_DATA_PATH):
        print(f"📂 Found custom dataset. Loading from: {CUSTOM_DATA_PATH}")
        df = pd.read_csv(CUSTOM_DATA_PATH)
    else:
        print(
            f"🌐 No custom file found. Downloading official dataset: {DEFAULT_DATA_URL}"
        )
        try:
            df = pd.read_csv(DEFAULT_DATA_URL)
            print("✅ Download successful.")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return

    print(f"📊 Raw data shape: {df.shape}")

    df = df.dropna()

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    if "penguin_id" not in df.columns:
        df["penguin_id"] = ["P" + str(i + 1000) for i in range(len(df))]

    print(f"🧹 Cleaned data shape: {df.shape}")

    print("🚀 Ingesting into MongoDB...")
    mongo_coll = get_mongo_collection()
    mongo_coll.delete_many({})

    mongo_docs = []
    for _, row in df.iterrows():
        doc = {
            "penguin_id": row["penguin_id"],
            "species": row["species"],
            "island": row["island"],
            "sex": row.get("sex", "unknown"),
            "year": int(row.get("year", 2020)),
            "features": {
                "bill_length": row.get("bill_length_mm"),
                "bill_depth": row.get("bill_depth_mm"),
                "flipper_length": row.get("flipper_length_mm"),
                "body_mass": row.get("body_mass_g"),
            },
            "label": row["species"],
        }
        mongo_docs.append(doc)

    if mongo_docs:
        mongo_coll.insert_many(mongo_docs)
        print(f"✅ MongoDB: Inserted {len(mongo_docs)} documents.")

    print("🚀 Ingesting into Cassandra...")
    session = get_cassandra_session()

    session.execute("DROP TABLE IF EXISTS penguins_by_island;")
    session.execute("""
        CREATE TABLE penguins_by_island (
            island text,
            species text,
            penguin_id text,
            bill_length float,
            bill_depth float,
            flipper_length int,
            body_mass int,
            prediction double,
            PRIMARY KEY ((island), species, penguin_id)
        );
    """)

    prepared = session.prepare("""
        INSERT INTO penguins_by_island (island, species, penguin_id, bill_length, bill_depth, flipper_length, body_mass)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """)

    count = 0
    for _, row in df.iterrows():
        try:
            session.execute(
                prepared,
                (
                    row["island"],
                    row["species"],
                    row["penguin_id"],
                    float(row["bill_length_mm"]),
                    float(row["bill_depth_mm"]),
                    int(row["flipper_length_mm"]),
                    int(row["body_mass_g"]),
                ),
            )
            count += 1
        except Exception as e:
            print(f"⚠️ Error inserting row {row['penguin_id']}: {e}")

    print(f"✅ Cassandra: Inserted {count} rows.")
    print("--- Ingestion Complete ---")


if __name__ == "__main__":
    ingest_data()
