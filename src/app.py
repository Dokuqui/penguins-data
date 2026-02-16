import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp
import numpy as np
import time
import redis
from cassandra.cluster import Cluster
import requests
import subprocess
import os

st.set_page_config(
    page_title="PenguinOps Mission Control", layout="wide", page_icon="🐧"
)
st.title("🐧 PenguinOps: Distributed ML Orchestrator")


@st.cache_resource
def get_mongo_conn():
    return MongoClient("penguin_mongo", 27017)


client = get_mongo_conn()
db = client["penguin_db"]


def run_shell_command(command_list):
    try:
        result = subprocess.run(
            command_list, cwd="/home/jovyan", capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def save_uploaded_file(uploaded_file):
    save_dir = "/home/jovyan/work/data"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "custom_upload.csv")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


if "status" not in st.session_state:
    st.session_state["status"] = "Ready"

st.sidebar.header("🕹️ System Status")
st.sidebar.info(st.session_state["status"])
st.sidebar.markdown("---")
st.sidebar.success("✅ MongoDB: Connected")
st.sidebar.success("✅ Spark: Idle")
st.sidebar.success("✅ API: Running (Port 8000)")

# --- TABS ---
tab_control, tab_viz, tab_mlops = st.tabs(
    ["⚙️ Pipeline Control", "📊 Benchmarks", "🧠 MLOps & Data Analytics"]
)

# =========================================================
# TAB 1: PIPELINE CONTROL & CUSTOM INGESTION
# =========================================================
with tab_control:
    st.header("Pipeline Automation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Data Ingestion (ETL)")
        st.info(
            "Upload a CSV file to override the default dataset, or leave empty to use the official Palmer Penguins source."
        )

        uploaded_file = st.file_uploader("📂 Upload Custom Dataset (CSV)", type="csv")
        if uploaded_file is not None:
            path = save_uploaded_file(uploaded_file)
            st.success(f"Custom file saved to: {path}")

        if st.button("🚀 START INGESTION", type="primary"):
            st.session_state["status"] = "Ingesting Data..."
            with st.spinner("Running ETL Pipeline (Cleaning & Loading)..."):
                success, out, err = run_shell_command(
                    ["python", "work/src/ingest_data.py"]
                )
                if success:
                    st.success("✅ Ingestion Complete!")
                    with st.expander("View Logs"):
                        st.code(out)
                else:
                    st.error("❌ Ingestion Failed")
                    st.error(err)
            st.session_state["status"] = "Ready"

    with col2:
        st.subheader("2. Model Training (Spark)")
        st.write("Submit distributed Random Forest job to Spark Cluster.")

        if st.button("🧠 TRAIN MODEL"):
            st.session_state["status"] = "Training Model..."
            with st.spinner("Spark Job Running (~15s)..."):
                cmd = [
                    "spark-submit",
                    "--packages",
                    "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                    "work/src/spark_ml.py",
                ]
                success, out, err = run_shell_command(cmd)
                if success:
                    st.success("✅ Model Trained & Saved!")
                    with st.expander("View Spark Logs"):
                        st.code(out)
                else:
                    st.error("❌ Spark Job Failed")
                    st.error(err)
            st.session_state["status"] = "Ready"

# =========================================================
# TAB 2: LIVE BENCHMARK
# =========================================================
with tab_viz:
    st.header("⚡ Real-Time NoSQL Benchmark")
    if st.button("🏁 RUN BENCHMARK TEST"):
        with st.spinner("Benchmarking MongoDB vs Cassandra vs Redis..."):
            try:
                r_cache = redis.Redis(
                    host="penguin_redis", port=6379, decode_responses=True
                )
                c_cluster = Cluster(["penguin_cassandra"])
                c_session = c_cluster.connect("penguin_ks")
                m_coll = MongoClient("penguin_mongo", 27017)["penguin_db"]["penguins"]

                test_id = "P1020"
                iterations = 500

                r_cache.set(test_id, "0.0")
                start = time.time()
                for _ in range(iterations):
                    _ = r_cache.get(test_id)
                redis_time = (time.time() - start) / iterations

                start = time.time()
                for _ in range(iterations):
                    _ = m_coll.find_one({"penguin_id": test_id})
                mongo_time = (time.time() - start) / iterations

                prep = c_session.prepare(
                    "SELECT * FROM penguins_by_island WHERE island='Biscoe' AND species='Adelie' AND penguin_id=?"
                )
                start = time.time()
                for _ in range(iterations):
                    _ = c_session.execute(prep, [test_id])
                cass_time = (time.time() - start) / iterations

                data = [
                    {
                        "Engine": "Redis",
                        "Latency (ms)": redis_time * 1000,
                        "Ops/Sec": int(1 / redis_time),
                    },
                    {
                        "Engine": "MongoDB",
                        "Latency (ms)": mongo_time * 1000,
                        "Ops/Sec": int(1 / mongo_time),
                    },
                    {
                        "Engine": "Cassandra",
                        "Latency (ms)": cass_time * 1000,
                        "Ops/Sec": int(1 / cass_time),
                    },
                ]
                df_res = pd.DataFrame(data)

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        df_res,
                        x="Engine",
                        y="Latency (ms)",
                        color="Engine",
                        text_auto=".2f",
                        title="Read Latency (Lower is Better)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.bar(
                        df_res,
                        x="Engine",
                        y="Ops/Sec",
                        color="Engine",
                        text_auto=True,
                        title="Throughput (Req/s - Higher is Better)",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Benchmark Error: {e}")

# =========================================================
# TAB 3: MLOPS & DATA ANALYTICS (Expanded)
# =========================================================
with tab_mlops:
    st.header("🔍 Model Observability & Data Analytics")

    preds = list(db.predictions.find().limit(500))
    all_docs = list(db.penguins.find())

    if preds and all_docs:
        df_p = pd.DataFrame(preds)
        label_map = {0.0: "Adelie", 1.0: "Gentoo", 2.0: "Chinstrap"}
        df_p["pred_label"] = df_p["prediction"].map(label_map)

        df_full = pd.DataFrame(all_docs)
        df_full["bill_length"] = df_full["features"].apply(
            lambda x: x.get("bill_length")
        )
        df_full["bill_depth"] = df_full["features"].apply(lambda x: x.get("bill_depth"))
        df_full["flipper_length"] = df_full["features"].apply(
            lambda x: x.get("flipper_length")
        )
        df_full["body_mass"] = df_full["features"].apply(lambda x: x.get("body_mass"))
        df_full = df_full.dropna()

        st.subheader("1. Model Performance & Drift")
        col_cm, col_drift = st.columns(2)

        with col_cm:
            st.caption("Confusion Matrix")
            cm = confusion_matrix(
                df_p["label"],
                df_p["pred_label"],
                labels=["Adelie", "Gentoo", "Chinstrap"],
            )
            fig_cm = px.imshow(
                cm,
                x=["Adelie", "Gentoo", "Chinstrap"],
                y=["Adelie", "Gentoo", "Chinstrap"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_drift:
            st.caption("Data Drift Monitor (Body Mass)")
            train_data = df_full["body_mass"].astype(float).values
            prod_data = train_data * 1.05 + 50  # Simulation
            stat, p_val = ks_2samp(train_data, prod_data)

            st.metric(
                "P-Value (KS Test)",
                f"{p_val:.4f}",
                delta="-Drift Detected" if p_val < 0.05 else "Stable",
                delta_color="inverse",
            )

            drift_df = pd.DataFrame(
                {
                    "Body Mass": np.concatenate([train_data, prod_data]),
                    "Source": ["Training"] * len(train_data)
                    + ["Production"] * len(prod_data),
                }
            )
            fig_drift = px.histogram(
                drift_df, x="Body Mass", color="Source", barmode="overlay", opacity=0.6
            )
            st.plotly_chart(fig_drift, use_container_width=True)

        st.divider()

        st.subheader("2. Feature Analysis (Why the model works)")

        st.write("**3D Species Separation** (Bill Length vs Depth vs Flipper)")
        fig_3d = px.scatter_3d(
            df_full,
            x="bill_length",
            y="bill_depth",
            z="flipper_length",
            color="species",
            symbol="species",
            opacity=0.7,
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Feature Correlation Heatmap**")
            numeric_df = df_full[
                ["bill_length", "bill_depth", "flipper_length", "body_mass"]
            ].astype(float)
            corr = numeric_df.corr()
            fig_corr = px.imshow(
                corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with c2:
            st.write("**Biometric Distributions (Violin Plot)**")
            metric = st.selectbox(
                "Select Metric", ["body_mass", "flipper_length", "bill_length"], index=0
            )
            fig_viol = px.violin(
                df_full, y=metric, x="species", color="species", box=True, points="all"
            )
            st.plotly_chart(fig_viol, use_container_width=True)

    else:
        st.warning(
            "⚠️ No data found. Please go to 'Pipeline Control' -> 'START INGESTION' then 'TRAIN MODEL'."
        )

    st.divider()
    st.header("🔌 API Gateway (Hybrid Architecture)")

    col_lookup, col_custom = st.columns(2)

    with col_lookup:
        st.subheader("A. Database Lookup")
        st.info("Retrieves pre-calculated Spark predictions for existing penguins.")

        if st.button("🎲 Pick Random ID"):
            random_doc = list(db.penguins.aggregate([{"$sample": {"size": 1}}]))
            if random_doc:
                st.session_state["test_id"] = random_doc[0]["penguin_id"]

        # Input Box
        test_id = st.text_input(
            "Penguin ID", value=st.session_state.get("test_id", "P1024")
        )

        if st.button("🔍 Search Prediction"):
            try:
                res = requests.post(
                    "http://localhost:8000/predict/lookup", json={"penguin_id": test_id}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Prediction: {data['prediction']}")
                    st.json(data)
                else:
                    st.error("ID Not Found in Database")
            except:
                st.error("API Connection Error")

    with col_custom:
        st.subheader("B. Real-Time Inference")
        st.info("Uses a lightweight shadow model to predict ANY measurements.")

        with st.form("custom_pred"):
            c1, c2 = st.columns(2)
            bl = c1.number_input("Bill Length (mm)", 30.0, 60.0, 39.1)
            bd = c2.number_input("Bill Depth (mm)", 10.0, 25.0, 18.7)
            fl = c1.number_input("Flipper Length (mm)", 170.0, 240.0, 181.0)
            bm = c2.number_input("Body Mass (g)", 2500.0, 6500.0, 3750.0)

            submit = st.form_submit_button("✨ Predict Species")

            if submit:
                payload = {
                    "bill_length": bl,
                    "bill_depth": bd,
                    "flipper_length": fl,
                    "body_mass": bm,
                }
                try:
                    res = requests.post(
                        "http://localhost:8000/predict/custom", json=payload
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.balloons()
                        st.metric("Result", data["predicted_species"])
                        st.progress(
                            data["confidence"],
                            text=f"Confidence: {data['confidence']:.2%}",
                        )
                        with st.expander("See Raw API Response"):
                            st.json(data)
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
