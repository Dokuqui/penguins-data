from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from contextlib import asynccontextmanager

MONGO_HOST = "penguin_mongo"
REDIS_HOST = "penguin_redis"

models = {}
db = None
cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, cache
    try:
        db = MongoClient(MONGO_HOST, 27017)["penguin_db"]
        cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        print("✅ API: Databases Connected")
    except Exception as e:
        print(f"❌ API: Database Connection Failed: {e}")

    print("🧠 API: Training Real-time Shadow Model...")
    try:
        data = list(db.penguins.find({}, {"_id": 0}))
        if data:
            df = pd.DataFrame(data)
            df["bill_length"] = df["features"].apply(lambda x: x.get("bill_length"))
            df["bill_depth"] = df["features"].apply(lambda x: x.get("bill_depth"))
            df["flipper_length"] = df["features"].apply(
                lambda x: x.get("flipper_length")
            )
            df["body_mass"] = df["features"].apply(lambda x: x.get("body_mass"))

            df = df.dropna()
            X = df[["bill_length", "bill_depth", "flipper_length", "body_mass"]]
            y = df["species"]

            clf = RandomForestClassifier(n_estimators=20, max_depth=5)
            clf.fit(X, y)
            models["rf_shadow"] = clf
            print("✅ API: Shadow Model Trained & Ready!")
        else:
            print("⚠️ API: No data found in Mongo. Run Ingestion first.")
    except Exception as e:
        print(f"❌ API: Model training failed: {e}")

    yield
    models.clear()


app = FastAPI(title="Penguin Prediction Service", lifespan=lifespan)


class PenguinLookup(BaseModel):
    penguin_id: str


class CustomFeatures(BaseModel):
    bill_length: float
    bill_depth: float
    flipper_length: float
    body_mass: float


@app.get("/")
def health():
    return {"status": "online", "model_ready": "rf_shadow" in models}


@app.post("/predict/lookup")
def predict_by_id(lookup: PenguinLookup):
    p_id = lookup.penguin_id

    if cache.exists(p_id):
        return {
            "mode": "lookup",
            "penguin_id": p_id,
            "prediction": cache.get(p_id),
            "source": "Redis (Hot Path)",
        }

    doc = db.predictions.find_one({"penguin_id": p_id})
    if doc:
        pred = doc["prediction"]
        cache.setex(p_id, 3600, str(pred))
        return {
            "mode": "lookup",
            "penguin_id": p_id,
            "prediction": pred,
            "source": "MongoDB (Cold Path)",
        }

    raise HTTPException(404, "Penguin ID not found")


@app.post("/predict/custom")
def predict_custom(features: CustomFeatures):
    if "rf_shadow" not in models:
        raise HTTPException(503, "Model not trained yet. Please run Ingestion.")

    X_input = pd.DataFrame([features.dict()])

    model = models["rf_shadow"]
    pred_class = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input).max()

    return {
        "mode": "inference",
        "input": features.dict(),
        "predicted_species": pred_class,
        "confidence": float(pred_prob),
        "engine": "Scikit-Learn (Shadow Model)",
    }
