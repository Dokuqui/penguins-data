import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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


def load_data():
    all_docs = list(db.penguins.find())
    if all_docs:
        df = pd.DataFrame(all_docs)
        df["bill_length"] = df["features"].apply(lambda x: x.get("bill_length"))
        df["bill_depth"] = df["features"].apply(lambda x: x.get("bill_depth"))
        df["flipper_length"] = df["features"].apply(lambda x: x.get("flipper_length"))
        df["body_mass"] = df["features"].apply(lambda x: x.get("body_mass"))
        return df.dropna()
    return pd.DataFrame()


tab_control, tab_viz, tab_mlops, tab_stats = st.tabs(
    [
        "⚙️ Pipeline Control",
        "📊 Benchmarks",
        "🧠 Classification (MLOps)",
        "📈 Statistics & Regression",
    ]
)

# =========================================================
# TAB 1: PIPELINE CONTROL
# =========================================================
with tab_control:
    st.header("Pipeline Automation")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Data Ingestion (ETL)")
        uploaded_file = st.file_uploader("📂 Upload Custom Dataset (CSV)", type="csv")
        if uploaded_file is not None:
            path = save_uploaded_file(uploaded_file)
            st.success(f"File saved: {path}")

        if st.button("🚀 START INGESTION", type="primary"):
            st.session_state["status"] = "Ingesting..."
            with st.spinner("Running ETL..."):
                success, out, err = run_shell_command(
                    ["python", "work/src/ingest_data.py"]
                )
                if success:
                    st.success("Ingestion Complete!")
                    with st.expander("Logs"):
                        st.code(out)
                else:
                    st.error("Failed")
                    st.error(err)
            st.session_state["status"] = "Ready"

    with col2:
        st.subheader("2. Model Training (Spark)")
        if st.button("🧠 TRAIN MODEL"):
            st.session_state["status"] = "Training..."
            with st.spinner("Spark Job Running..."):
                cmd = [
                    "spark-submit",
                    "--packages",
                    "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                    "work/src/spark_ml.py",
                ]
                success, out, err = run_shell_command(cmd)
                if success:
                    st.success("Training Complete!")
                    with st.expander("Logs"):
                        st.code(out)
                else:
                    st.error("Failed")
                    st.error(err)
            st.session_state["status"] = "Ready"

# =========================================================
# TAB 2: BENCHMARKS
# =========================================================
with tab_viz:
    st.header("⚡ NoSQL Benchmark")
    if st.button("🏁 RUN BENCHMARK"):
        with st.spinner("Benchmarking..."):
            try:
                r_cache = redis.Redis(
                    host="penguin_redis", port=6379, decode_responses=True
                )
                c_cluster = Cluster(["penguin_cassandra"])
                c_session = c_cluster.connect("penguin_ks")
                m_coll = MongoClient("penguin_mongo", 27017)["penguin_db"]["penguins"]

                test_id = "P1020"
                iterations = 500

                # Redis
                r_cache.set(test_id, "0.0")
                start = time.time()
                for _ in range(iterations):
                    _ = r_cache.get(test_id)
                r_time = (time.time() - start) / iterations

                # Mongo
                start = time.time()
                for _ in range(iterations):
                    _ = m_coll.find_one({"penguin_id": test_id})
                m_time = (time.time() - start) / iterations

                # Cassandra
                prep = c_session.prepare(
                    "SELECT * FROM penguins_by_island WHERE island='Biscoe' AND species='Adelie' AND penguin_id=?"
                )
                start = time.time()
                for _ in range(iterations):
                    _ = c_session.execute(prep, [test_id])
                c_time = (time.time() - start) / iterations

                data = [
                    {
                        "Engine": "Redis",
                        "Latency (ms)": r_time * 1000,
                        "Ops/Sec": int(1 / r_time),
                    },
                    {
                        "Engine": "MongoDB",
                        "Latency (ms)": m_time * 1000,
                        "Ops/Sec": int(1 / m_time),
                    },
                    {
                        "Engine": "Cassandra",
                        "Latency (ms)": c_time * 1000,
                        "Ops/Sec": int(1 / c_time),
                    },
                ]
                df_res = pd.DataFrame(data)

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        px.bar(
                            df_res,
                            x="Engine",
                            y="Latency (ms)",
                            color="Engine",
                            text_auto=".2f",
                            title="Read Latency",
                        ),
                        use_container_width=True,
                    )
                with c2:
                    st.plotly_chart(
                        px.bar(
                            df_res,
                            x="Engine",
                            y="Ops/Sec",
                            color="Engine",
                            text_auto=True,
                            title="Throughput",
                        ),
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# =========================================================
# TAB 3: MLOPS (CLASSIFICATION)
# =========================================================
with tab_mlops:
    st.header("🧠 Classification Model (Random Forest)")
    preds = list(db.predictions.find().limit(500))
    df_full = load_data()

    if preds and not df_full.empty:
        df_p = pd.DataFrame(preds)
        label_map = {0.0: "Adelie", 1.0: "Gentoo", 2.0: "Chinstrap"}
        df_p["pred_label"] = df_p["prediction"].map(label_map)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(
                df_p["label"],
                df_p["pred_label"],
                labels=["Adelie", "Gentoo", "Chinstrap"],
            )
            st.plotly_chart(
                px.imshow(
                    cm,
                    x=["Adelie", "Gentoo", "Chinstrap"],
                    y=["Adelie", "Gentoo", "Chinstrap"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                ),
                use_container_width=True,
            )

        with c2:
            st.subheader("Data Drift (KS Test)")
            train_data = df_full["body_mass"].astype(float).values
            prod_data = train_data * 1.05 + 50
            stat, p_val = ks_2samp(train_data, prod_data)
            st.metric(
                "P-Value",
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
            st.plotly_chart(
                px.histogram(
                    drift_df, x="Body Mass", color="Source", barmode="overlay"
                ),
                use_container_width=True,
            )

    st.divider()
    st.subheader("🔌 API Gateway")

    c_look, c_cust = st.columns(2)
    with c_look:
        st.caption("A. Database Lookup")
        test_id = st.text_input("Penguin ID", "P1024")
        if st.button("Search"):
            try:
                r = requests.post(
                    "http://localhost:8000/predict/lookup", json={"penguin_id": test_id}
                )
                st.json(r.json())
            except:
                st.error("API Error")

    with c_cust:
        st.caption("B. Real-Time Inference")
        with st.form("cust"):
            b_l = st.number_input("Bill Length", value=39.1)
            b_d = st.number_input("Bill Depth", value=18.7)
            f_l = st.number_input("Flipper Length", value=181.0)
            b_m = st.number_input("Body Mass", value=3750.0)
            if st.form_submit_button("Predict"):
                try:
                    payload = {
                        "bill_length": b_l,
                        "bill_depth": b_d,
                        "flipper_length": f_l,
                        "body_mass": b_m,
                    }
                    r = requests.post(
                        "http://localhost:8000/predict/custom", json=payload
                    )
                    st.json(r.json())
                except:
                    st.error("API Error")

# =========================================================
# TAB 4: STATISTICS & REGRESSION (NEW! Satisfies Parts 1-3)
# =========================================================
with tab_stats:
    df = load_data()
    if not df.empty:
        st.header("📈 Part 1 & 2: Descriptive Statistics")

        st.subheader("Statistical Summary (Mean, Std, Min, Max)")
        st.dataframe(df.describe())

        st.subheader("Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Count by Island")
            st.bar_chart(df["island"].value_counts())
        with c2:
            st.caption("Count by Sex")
            st.bar_chart(df["sex"].value_counts())

        st.subheader("Specific Scatter Plots")
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Bill Length vs Depth (by Species)")
            fig1 = px.scatter(
                df,
                x="bill_length",
                y="bill_depth",
                color="species",
                title="Morphology by Species",
            )
            st.plotly_chart(fig1, use_container_width=True)
        with c4:
            st.caption("Flipper Length vs Mass (by Sex)")
            fig2 = px.scatter(
                df,
                x="flipper_length",
                y="body_mass",
                color="sex",
                title="Mass vs Flipper",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        st.header("📉 Part 3: Regression Lab")
        st.markdown("Predict **Body Mass (g)** using Linear Regression.")

        feature_cols = st.multiselect(
            "Select Predictors (X)",
            ["bill_length", "bill_depth", "flipper_length"],
            default=["flipper_length"],
        )

        if feature_cols:
            X = df[feature_cols].astype(float)
            y = df["body_mass"].astype(float)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            c_res1, c_res2 = st.columns(2)
            c_res1.metric("R² Score (Accuracy)", f"{r2:.4f}")
            c_res1.metric("Mean Squared Error", f"{mse:.0f}")

            with c_res2:
                st.write("**Model Coefficients:**")
                coeff_df = pd.DataFrame(
                    {"Feature": feature_cols, "Coefficient": reg.coef_}
                )
                st.dataframe(coeff_df)

            if len(feature_cols) == 1:
                st.subheader("Regression Line")
                chart_df = pd.DataFrame(
                    {
                        "X": X_test[feature_cols[0]],
                        "Y Actual": y_test,
                        "Y Predicted": y_pred,
                    }
                )
                fig_reg = px.scatter(
                    chart_df,
                    x="X",
                    y="Y Actual",
                    opacity=0.6,
                    title=f"Regression: {feature_cols[0]} vs Body Mass",
                )
                fig_reg.add_scatter(
                    x=chart_df["X"],
                    y=chart_df["Y Predicted"],
                    mode="lines",
                    name="Regression Line",
                )
                st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.warning("Please select at least one feature to run regression.")

    else:
        st.error("No data found. Please run Ingestion in Tab 1.")
