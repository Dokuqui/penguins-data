# 🐧 PenguinOps: Distributed Big Data & MLOps Platform

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-blue?style=for-the-badge&logo=docker&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Big%20Data-orange?style=for-the-badge&logo=apachespark&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Document%20Store-green?style=for-the-badge&logo=mongodb&logoColor=white)
![Cassandra](https://img.shields.io/badge/Cassandra-Wide%20Column-skyblue?style=for-the-badge&logo=apachecassandra&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Caching-red?style=for-the-badge&logo=redis&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**PenguinOps** is a production-grade **Lambda Architecture** designed to demonstrate the complete lifecycle of a Distributed Machine Learning system. It orchestrates data ingestion, distributed training (Spark), and real-time inference (FastAPI/Redis) to classify Palmer Penguin species.

---

## 🏗️ Architecture

The system follows a **Multi-NoSQL** approach to handle the 3 Vs of Big Data (Volume, Variety, Velocity).

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Ingestion** | **Python ETL** | Fetches raw CSV data, cleans it, and dispatches it to the Data Lake. |
| **Storage (Lake)** | **MongoDB** | Stores semi-structured JSON documents (flexible schema). |
| **Storage (Ware)** | **Cassandra** | Stores analytical data partitioned by geography (`island`) for high availability. |
| **Processing** | **Apache Spark** | Distributed training of Random Forest models (`Spark MLlib`) on the cluster. |
| **Speed Layer** | **Redis** | Caches high-frequency predictions (< 1ms latency). |
| **Serving** | **FastAPI** | Microservice exposing REST endpoints for real-time inference. |
| **Visualization** | **Streamlit** | Interactive Mission Control dashboard for MLOps & Analytics. |

---

## 🚀 Features

### 1. Hybrid Inference Engine

- **Batch Mode:** Spark processes historical data and saves results to Cassandra.
- **Real-Time Mode:** API trains a "Shadow Model" (Scikit-Learn) on startup to predict *unknown* penguins instantly.

### 2. MLOps Observability

- **Data Drift Detection:** Automatically monitors `body_mass` distribution using the **Kolmogorov-Smirnov Test**.
- **Performance Tracking:** Live Confusion Matrix comparing MongoDB labels vs. Spark predictions.

### 3. Advanced Benchmarking

- Built-in benchmarking tool comparing **Redis vs. MongoDB vs. Cassandra** read latencies.
- **Results:** Redis (~0.18ms) proves 8x faster than Cassandra for point lookups.

### 4. Interactive Data Lab

- **Regression Analysis:** Perform Linear Regression to predict body mass.
- **3D Visualization:** Explore species separation in 3D space (Bill Length x Depth x Flipper).

---

## 🛠️ Installation & Usage

### Prerequisites

- Docker & Docker Compose installed.
- 4GB+ RAM available (Spark requirement).

### 1. Clone the Repository

```bash
git clone [https://github.com/dokuqui/penguins-data.git](https://github.com/dokuqui/penguins-data.git)
cd penguins-data
```

### 2. Run with Docker Compose

```bash
docker-compose up -d --build

```

*Wait ~2 minutes for the Spark container to initialize and train the models.*

### 3. Access the Platform

- **📊 Dashboard:** [http://localhost:8501]()
- **🔌 API Docs:** [http://localhost:8000/docs]()
- **📓 Jupyter:** [http://localhost:8888]()

---

## ☁️ Deployment (VPS / Cloud)

This project includes a production-ready `deploy.sh` script for deploying to **AWS, Azure, or OVH**.

1. **Upload to VPS:**

```bash
scp -r * root@YOUR_VPS_IP:~/penguin_project/

```

1. **Run Deployment Script:**

```bash
ssh root@YOUR_VPS_IP
cd penguin_project
bash deploy.sh

```

*The script automatically installs Docker, optimizes memory limits, and launches the stack.*

---

## 📂 Project Structure

```text
├── docker-compose.prod.yml  # Production container orchestration
├── Dockerfile               # Custom Spark + Python image
├── deploy.sh                # Auto-deployment script
├── work/
│   ├── src/
│   │   ├── app.py           # Streamlit Dashboard (Frontend)
│   │   ├── api.py           # FastAPI Service (Backend)
│   │   ├── spark_ml.py      # Spark Training Job
│   │   ├── ingest_data.py   # ETL Pipeline
│   │   └── benchmark.py     # Database Stress Test
│   └── data/                # Local data storage
└── README.md                # Documentation

```
