from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time

from core.settings import Settings

settings = Settings()

app = FastAPI(title = "Barrikada")
MODEL_PATH = settings.model_path
VECTORIZER_PATH = settings.vectorizer_path

vec = joblib.load(VECTORIZER_PATH)
clf = joblib.load(MODEL_PATH)

class Req(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

#TODO: implement caching for ML inference
@app.post("/score")
def score(req: Req):
    s = req.prompt.strip()

    t0 = time.time()
    x = vec.transform([s])
    score = float(clf.predict_proba(x)[:,1])
    latency = (time.time() - t0) * 1000  #ms

    return {"score": score, "latency_ms": latency, "cached": False} #TODO: change cached to dynamic when caching is implemented


