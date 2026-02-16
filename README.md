# NoSQL Benchmarking & Penguin Classification with Spark MLlib

## Project Overview

This project implements a complete Big Data pipeline for the classification of penguin species using the Palmer Penguins dataset. It demonstrates a multi-NoSQL architecture using Document (MongoDB), Column-Family (Cassandra), and Key-Value (Redis) stores, integrated with Apache Spark for distributed machine learning and Streamlit for visual analytics.

## Architecture

* **Ingestion:** Python ETL script fetching raw CSV data and loading it into NoSQL engines.
* **Storage:** - **MongoDB:** Stores nested documents (features sub-document).
* **Cassandra:** Partitioned by `island` for distributed performance.

* **Processing:** **Spark MLlib** running a Random Forest Classifier with feature scaling.
* **Optimization:** **Redis** used as a high-speed caching layer for predictions.
* **Visualization:** **Streamlit Dashboard** providing MLOps metrics, confusion matrices, and performance charts.

---

## Prerequisites

* Docker & Docker Desktop
* Port **8501** available on your host machine

---

## Setup & Execution Guide

### 1. Launch Infrastructure

Start the NoSQL containers and the Spark environment:

```bash
docker-compose up -d

```

*Note: Wait ~60 seconds for Cassandra to fully initialize internal networking.*

### 2. Install Drivers inside Spark Container

The Spark container needs specific Python drivers to communicate with the databases and run the dashboard:

```bash
docker exec -it penguin_spark pip install pymongo cassandra-driver redis requests pandas pyspark streamlit plotly matplotlib seaborn scikit-learn

```

### 3. Data Ingestion (ETL)

Clean the data and load it into MongoDB and Cassandra:

```bash
docker exec -it penguin_spark python work/src/ingest_data.py

```

### 4. Distributed Machine Learning (Spark MLlib)

Prepare Cassandra schema and run the Spark job to train the model and save predictions:

```bash
docker exec -it penguin_cassandra cqlsh -e "DROP TABLE IF EXISTS penguin_ks.penguins_by_island;"
docker exec -it penguin_cassandra cqlsh -e "CREATE TABLE penguin_ks.penguins_by_island (island text, species text, penguin_id text, bill_length float, body_mass int, prediction double, PRIMARY KEY ((island), species, penguin_id));"

docker exec -it penguin_spark spark-submit \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0 \
  work/src/spark_ml.py

```

### 5. Launch Visual Dashboard (Streamlit)

Start the interactive dashboard to view MLOps metrics and NoSQL benchmarks:

```bash
docker exec -it penguin_spark python -m streamlit run work/src/app.py --server.address 0.0.0.0

```

**Access URL:** `http://localhost:8501`

---

## Benchmark Results (Actual Run)

| Technology | Avg Latency | Estimated Throughput |
| --- | --- | --- |
| MongoDB | 0.35 ms | 2823 req/s |
| Cassandra | 1.52 ms | 658 req/s |
| Redis | 0.18 ms | 5677 req/s |

## Key Findings

* **Redis Efficiency:** Proved to be **~8.5x faster** than Cassandra for point-lookups, justifying its role as a caching layer for high-concurrency predictions.
* **Model Accuracy:** Implementing a `StandardScaler` was critical to resolve classification bias caused by the high variance in biometric scales (grams vs. millimeters).
* **Scalability:** Cassandra's partitioning by `island` enables the system to scale horizontally by distributing geographic data across multiple nodes.
