from fastapi import FastAPI
from pydantic import BaseModel
import time

from core.settings import Settings
from core.layer_c.classifier import Classifier

settings = Settings()

app = FastAPI(title = "Barrikada")
MODEL_PATH = settings.model_path
VECTORIZER_PATH = settings.vectorizer_path
REDUCER_PATH = settings.reducer_path

clf = Classifier(
    vectorizer_path=VECTORIZER_PATH,
    model_path=MODEL_PATH,
    reducer_path=REDUCER_PATH,
    low=settings.layer_c_low_threshold,
    high=settings.layer_c_high_threshold,
)

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
    out = clf.predict_dict(s)
    latency = (time.time() - t0) * 1000  #ms

    return {"score": out["score"], "decision": out["decision"], "latency_ms": latency, "cached": False} #TODO: change cached to dynamic when caching is implemented


