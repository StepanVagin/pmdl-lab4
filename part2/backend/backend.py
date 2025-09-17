import uvicorn
from schemas.SampleSchema import SampleSchema
from fastapi import FastAPI
from fastapi import UploadFile, HTTPException
import os 

import joblib
import pandas as pd

app = FastAPI()
# In Docker: MODEL_DIR=/models, file is at /models/best_model_pipeline.joblib
# In GitHub Actions: use relative path ../models/best/best_model_pipeline.joblib
model_dir = os.getenv("MODEL_DIR")
if model_dir:
    model_path = os.path.join(model_dir, "best_model_pipeline.joblib")
else:
    model_path = "../models/best/best_model_pipeline.joblib"
model = joblib.load(model_path)

@app.get("/health", tags=['health'])  
def health_check():
    return {"status": "healthy"}
  
@app.post("/predict", tags=['predict'])
def predict(sample: SampleSchema):
    df = pd.DataFrame([dict(sample)])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
  
@app.post("/predict_batch", tags=['predict'])
def predict_batch(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, detail="Only CSV files are allowed.")
    
    df = pd.read_csv(file.file, encoding='utf-8', sep=';')    
    predictions = model.predict(df)
    
    return {
        "predictions": predictions.tolist()
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)