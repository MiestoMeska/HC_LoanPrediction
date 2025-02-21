from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from joblib import load
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), '..', 'lgbm_best_model.joblib')
model = load(model_path)

features_path = os.path.join(os.path.dirname(__file__), '../../data', 'selected_features.csv')
selected_features = pd.read_csv(features_path).squeeze("columns").tolist()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(content={"message": "This API supports only CSV files."}, status_code=400)
    
    df = pd.read_csv(file.file)
    
    if 'SK_ID_CURR' not in df.columns:
        return JSONResponse(content={"message": "CSV must contain 'SK_ID_CURR' column."}, status_code=400)
    
    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        return JSONResponse(content={"message": f"Missing features in the CSV: {missing_features}"}, status_code=400)

    filtered_df = df[selected_features]

    try:
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan)
        predictions = model.predict(filtered_df)
        prediction_probs = model.predict_proba(filtered_df)[:, 1]
        
        df['TARGET'] = predictions
        df['TARGET_PROB'] = prediction_probs
        
        df.fillna("null", inplace=True)
        
        result = df[['SK_ID_CURR', 'TARGET', 'TARGET_PROB']].to_dict(orient="records")
        return JSONResponse(content={"predictions": result})
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)


#uvicorn trained_models.deployment.app_API:app --reload