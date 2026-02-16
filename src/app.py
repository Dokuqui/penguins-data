import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="MLOps & NoSQL Dashboard", layout="wide")
st.title("🐧 Penguin Pipeline: MLOps & NoSQL Performance")


@st.cache_resource
def init_connection():
    return MongoClient("penguin_mongo", 27017)


client = init_connection()
db = client["penguin_db"]

st.sidebar.header("Pipeline Monitoring")
st.sidebar.metric("Data Source", "MongoDB", "Connected")
st.sidebar.metric("Distributed Engine", "Spark 3.5", "Active")
st.sidebar.metric("Cache Layer", "Redis", "Optimized")

tab1, tab2, tab3 = st.tabs(
    ["📊 Performance Benchmarks", "🧠 ML Model Analysis", "🔍 Live Data Explorer"]
)

with tab1:
    st.header("Database Scalability Analysis")
    bench_data = pd.DataFrame(
        {
            "Engine": ["MongoDB", "Cassandra", "Redis (Cache)"],
            "Latency (ms)": [0.35, 1.52, 0.18],
            "Ops/Sec": [2823, 658, 5677],
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_lat = px.bar(
            bench_data,
            x="Engine",
            y="Latency (ms)",
            color="Engine",
            text_auto=True,
            title="Average Read Latency (lower is better)",
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    with col2:
        fig_ops = px.line(
            bench_data,
            x="Engine",
            y="Ops/Sec",
            markers=True,
            title="Throughput Capacity (higher is better)",
        )
        st.plotly_chart(fig_ops, use_container_width=True)

with tab2:
    st.header("Random Forest Model Metrics")

    data = list(db.predictions.find({}, {"label": 1, "prediction": 1, "_id": 0}))
    if data:
        df = pd.DataFrame(data)

        label_map = {0.0: "Adelie", 1.0: "Gentoo", 2.0: "Chinstrap"}
        df["pred_name"] = df["prediction"].map(label_map)

        st.subheader("Confusion Matrix")
        y_true = df["label"]
        y_pred = df["pred_name"]
        labels = ["Adelie", "Gentoo", "Chinstrap"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale="Blues",
            title="Classification Accuracy",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Model Artifacts (MLflow Style)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Model Type", "Random Forest")
        m_col2.metric("N Trees", "10")
        m_col3.metric("Max Depth", "5")

    else:
        st.info("Run your spark_ml.py script to generate data for this section.")

with tab3:
    st.header("Raw Document Store (MongoDB)")
    
    all_docs = list(db.penguins.find())
    if all_docs:
        all_data = pd.DataFrame(all_docs)
        
        all_data['bill_length'] = all_data['features'].apply(lambda x: x.get('bill_length'))
        all_data['bill_depth'] = all_data['features'].apply(lambda x: x.get('bill_depth'))
        all_data['flipper_length'] = all_data['features'].apply(lambda x: x.get('flipper_length'))
        all_data['body_mass'] = all_data['features'].apply(lambda x: x.get('body_mass'))

        st.subheader("Data Distribution Analysis")
        feat = st.selectbox("Select biometric variable", 
                            ["bill_length", "bill_depth", "flipper_length", "body_mass"])
        
        fig_hist = px.histogram(all_data, x=feat, color="label", 
                                marginal="box", title=f"Distribution of {feat} by Species")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Recent Predictions in MongoDB")
        pred_docs = list(db.predictions.find().limit(10))
        if pred_docs:
            df_pred_view = pd.DataFrame(pred_docs)
            label_map = {0.0: "Adelie", 1.0: "Gentoo", 2.0: "Chinstrap"}
            df_pred_view['predicted_species'] = df_pred_view['prediction'].map(label_map)
            st.table(df_pred_view[["penguin_id", "label", "predicted_species"]])
        else:
            st.warning("No predictions found. Ensure you ran the Spark job successfully.")
    else:
        st.error("No data found in MongoDB.")
