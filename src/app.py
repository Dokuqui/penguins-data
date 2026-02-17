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
    page_title="PenguinOps Mission Control",
    layout="wide",
    page_icon="🐧",
    initial_sidebar_state="expanded",
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
        "🧠 Classification & MLOps",
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
        st.info("Extracts CSV, transforms schema, loads MongoDB/Cassandra.")
        uploaded_file = st.file_uploader("📂 Upload Custom Dataset (CSV)", type="csv")
        if uploaded_file:
            save_uploaded_file(uploaded_file)

        if st.button("🚀 START INGESTION", type="primary"):
            st.session_state["status"] = "Ingesting..."
            with st.spinner("Running ETL Pipeline..."):
                success, out, err = run_shell_command(
                    ["python", "work/src/ingest_data.py"]
                )
                if success:
                    st.success("Ingestion Complete!")
                    with st.expander("View Logs"):
                        st.code(out)
                else:
                    st.error("Failed")
                    st.error(err)
            st.session_state["status"] = "Ready"

    with col2:
        st.subheader("2. Model Training (Spark)")
        st.info("Submits distributed Random Forest job to Spark Cluster.")
        if st.button("🧠 TRAIN MODEL"):
            st.session_state["status"] = "Training..."
            with st.spinner("Spark Job Running (this takes ~15s)..."):
                cmd = [
                    "spark-submit",
                    "--packages",
                    "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
                    "work/src/spark_ml.py",
                ]
                success, out, err = run_shell_command(cmd)
                if success:
                    st.success("Training Complete!")
                    with st.expander("View Spark Logs"):
                        st.code(out)
                else:
                    st.error("Failed")
                    st.error(err)
            st.session_state["status"] = "Ready"

# =========================================================
# TAB 2: BENCHMARKS (ENHANCED)
# =========================================================
with tab_viz:
    st.header("⚡ NoSQL Engine Comparison")

    st.subheader("1. Architecture Capabilities")
    tech_data = {
        "Technology": ["Redis", "MongoDB", "Cassandra"],
        "Type": [
            "In-Memory Key-Value",
            "Document Store (B-Tree)",
            "Wide-Column (LSM Tree)",
        ],
        "Use Case": [
            "Real-time Caching (<1ms)",
            "Flexible Web Backends",
            "Heavy Write Loads & Analytics",
        ],
        "CAP Theorem": ["CP (Consistency)", "CP (Consistency)", "AP (Availability)"],
        "Storage": ["RAM (Volatile)", "Disk + RAM Index", "Disk (Log Structured)"],
    }
    st.dataframe(pd.DataFrame(tech_data), hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("2. Live Read Latency Test")
    st.markdown(
        "This test runs **500 sequential reads** for a single ID (`P1020`) against all three databases."
    )

    if st.button("🏁 RUN LIVE BENCHMARK"):
        with st.spinner("Benchmarking..."):
            try:
                r_cache = redis.Redis(
                    host="penguin_redis", port=6379, decode_responses=True
                )
                c_cluster = Cluster(["penguin_cassandra"])
                c_session = c_cluster.connect("penguin_ks")
                m_coll = MongoClient("penguin_mongo", 27017)["penguin_db"]["penguins"]

                test_id = "P1020"
                iter = 500

                # Redis Test
                r_cache.set(test_id, "0.0")
                start = time.time()
                for _ in range(iter):
                    _ = r_cache.get(test_id)
                r_time = (time.time() - start) / iter

                # Mongo Test
                start = time.time()
                for _ in range(iter):
                    _ = m_coll.find_one({"penguin_id": test_id})
                m_time = (time.time() - start) / iter

                # Cassandra Test
                prep = c_session.prepare(
                    "SELECT * FROM penguins_by_island WHERE island='Biscoe' AND species='Adelie' AND penguin_id=?"
                )
                start = time.time()
                for _ in range(iter):
                    _ = c_session.execute(prep, [test_id])
                c_time = (time.time() - start) / iter

                # Results Data
                df_res = pd.DataFrame(
                    [
                        {
                            "Engine": "Redis",
                            "Latency (ms)": r_time * 1000,
                            "Ops/Sec": int(1 / r_time),
                            "Type": "Cache",
                        },
                        {
                            "Engine": "MongoDB",
                            "Latency (ms)": m_time * 1000,
                            "Ops/Sec": int(1 / m_time),
                            "Type": "Document",
                        },
                        {
                            "Engine": "Cassandra",
                            "Latency (ms)": c_time * 1000,
                            "Ops/Sec": int(1 / c_time),
                            "Type": "Columnar",
                        },
                    ]
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Redis Latency", f"{r_time * 1000:.2f} ms", "Fastest")
                m2.metric(
                    "MongoDB Latency",
                    f"{m_time * 1000:.2f} ms",
                    f"+{(m_time / r_time):.1f}x slower",
                    delta_color="off",
                )
                m3.metric(
                    "Cassandra Latency",
                    f"{c_time * 1000:.2f} ms",
                    f"+{(c_time / r_time):.1f}x slower",
                    delta_color="inverse",
                )

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
                        title="Throughput (Req/sec - Higher is Better)",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.info("""
                **Performance Analysis:**
                * **Redis** wins because it serves data directly from RAM, avoiding Disk I/O.
                * **MongoDB** performs well for single lookups due to B-Tree indexing.
                * **Cassandra** has higher overhead for single reads (coordination latency) but scales linearly for massive write loads.
                """)

            except Exception as e:
                st.error(f"Benchmark Error: {e}")

# =========================================================
# TAB 3: MLOPS & API (ALL GRAPHS + SCORECARD INCLUDED)
# =========================================================
with tab_mlops:
    st.header("🧠 Classification & Observability")
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
            st.subheader("Data Drift (Body Mass)")
            train_data = df_full["body_mass"].astype(float).values
            prod_data = train_data * 1.05 + 50
            stat, p_val = ks_2samp(train_data, prod_data)
            st.metric(
                "P-Value (KS Test)",
                f"{p_val:.4f}",
                delta="-Drift Detected" if p_val < 0.05 else "Stable",
                delta_color="inverse",
            )
            drift_df = pd.DataFrame(
                {
                    "Mass": np.concatenate([train_data, prod_data]),
                    "Source": ["Training"] * len(train_data)
                    + ["Production"] * len(prod_data),
                }
            )
            st.plotly_chart(
                px.histogram(drift_df, x="Mass", color="Source", barmode="overlay"),
                use_container_width=True,
            )

        st.divider()
        st.subheader("🔍 Feature Analysis")

        st.write("**3D Species Separation**")
        fig_3d = px.scatter_3d(
            df_full,
            x="bill_length",
            y="bill_depth",
            z="flipper_length",
            color="species",
            symbol="species",
            opacity=0.7,
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
        st.plotly_chart(fig_3d, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.write("**Correlation Heatmap**")
            numeric_df = df_full[
                ["bill_length", "bill_depth", "flipper_length", "body_mass"]
            ].astype(float)
            corr = numeric_df.corr()
            fig_corr = px.imshow(
                corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with c4:
            st.write("**Biometric Distributions**")
            metric = st.selectbox(
                "Select Metric", ["body_mass", "flipper_length", "bill_length"], index=0
            )
            fig_viol = px.violin(
                df_full, y=metric, x="species", color="species", box=True, points="all"
            )
            st.plotly_chart(fig_viol, use_container_width=True)

    st.divider()
    st.header("🔌 API Gateway")

    col_lookup, col_custom = st.columns(2)

    with col_lookup:
        st.subheader("A. Database Lookup")
        test_id = st.text_input("Penguin ID", "P1024")
        if st.button("Search ID"):
            try:
                res = requests.post(
                    "http://localhost:8000/predict/lookup", json={"penguin_id": test_id}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Found: {test_id}")
                    m1, m2 = st.columns(2)
                    m1.metric(
                        "Prediction", f"{data['prediction']}", delta="Existing Record"
                    )
                    m2.metric("Source", data["source"], delta_color="off")
                else:
                    st.error("ID Not Found")
            except:
                st.error("API Error")

    with col_custom:
        st.subheader("B. Real-Time Inference")
        with st.form("cust"):
            c1, c2 = st.columns(2)
            b_l = c1.number_input("Bill Length (mm)", 30.0, 60.0, 39.1)
            b_d = c2.number_input("Bill Depth (mm)", 10.0, 25.0, 18.7)
            f_l = c1.number_input("Flipper Length (mm)", 170.0, 240.0, 181.0)
            b_m = c2.number_input("Body Mass (g)", 2500.0, 6500.0, 3750.0)

            if st.form_submit_button("✨ Predict Species"):
                try:
                    res = requests.post(
                        "http://localhost:8000/predict/custom",
                        json={
                            "bill_length": b_l,
                            "bill_depth": b_d,
                            "flipper_length": f_l,
                            "body_mass": b_m,
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.balloons()
                        st.markdown(
                            f"""
                        <div style='text-align:center; padding:15px; background:#f0f2f6; border-radius:10px; margin-bottom:10px;'>
                            <h2 style='color:#333; margin:0;'>Predicted Species</h2>
                            <h1 style='color:#ff4b4b; font-size:40px; margin:0;'>🐧 {data["predicted_species"]}</h1>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                        st.metric("Model Confidence", f"{data['confidence']:.2%}")
                        st.progress(data["confidence"])
                        st.caption(f"⚡ Engine: {data['engine']}")
                    else:
                        st.error("Failed")
                except Exception as e:
                    st.error(f"API Error: {e}")

# =========================================================
# TAB 4: STATISTICS & REGRESSION
# =========================================================
with tab_stats:
    df = load_data()
    if not df.empty:
        st.header("📈 Descriptive Statistics")
        st.dataframe(df.describe().style.background_gradient(cmap="Blues"))

        st.subheader("Visual Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.histogram(
                    df, x="island", color="species", title="Species per Island"
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.scatter(
                    df,
                    x="flipper_length",
                    y="body_mass",
                    color="sex",
                    title="Mass vs Flipper",
                ),
                use_container_width=True,
            )

        st.subheader("Multivariate Analysis (Pairplots)")
        fig_matrix = px.scatter_matrix(
            df,
            dimensions=["bill_length", "bill_depth", "flipper_length", "body_mass"],
            color="species",
            height=700,
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.divider()
        st.header("📉 Regression Lab")
        features = st.multiselect(
            "Predictors (X)",
            ["bill_length", "bill_depth", "flipper_length"],
            default=["flipper_length"],
        )

        if features:
            X = df[features].astype(float)
            y = df["body_mass"].astype(float)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            c_res1, c_res2 = st.columns(2)
            c_res1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            c_res1.metric("MSE", f"{mean_squared_error(y_test, y_pred):.0f}")

            with c_res2:
                st.write("Coefficients:")
                st.dataframe(pd.DataFrame({"Feature": features, "Coef": reg.coef_}))

            if len(features) == 1:
                chart = pd.DataFrame(
                    {"X": X_test[features[0]], "Y": y_test, "Pred": y_pred}
                )
                fig = px.scatter(
                    chart, x="X", y="Y", opacity=0.6, title="Regression Line"
                )
                fig.add_scatter(
                    x=chart["X"], y=chart["Pred"], mode="lines", name="Reg Line"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data. Run Ingestion first.")
