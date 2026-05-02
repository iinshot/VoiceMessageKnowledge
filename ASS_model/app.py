import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from scorer import StudentAnswerScorer

app = FastAPI(title="Answer Scoring Service")

class ScoreRequest(BaseModel):
    question: str
    reference: str
    student: str
    verbose: bool = False

def get_scorer():
    if not hasattr(get_scorer, "_instance"):
        model_dir = os.getenv("MODEL_DIR", "/models/siamese_rubert_s_128")
        get_scorer._instance = StudentAnswerScorer(
            model_dir=model_dir,
            weights={"w1": 0.8, "w2": 0.1, "w3": 0.1},
        )
    return get_scorer._instance

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(request: ScoreRequest):
    try:
        scorer = get_scorer()
        result = scorer.score(
            question=request.question,
            reference=request.reference,
            student=request.student,
            verbose=request.verbose,
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)