from fastapi import FastAPI, status, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pickle
import json
import pandas as pd

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources on startup
    print("Loading model and demographic data...")
    
    # Load the model
    with open("model/model.pkl", 'rb') as model_file:
        app.state.model = pickle.load(model_file)
    
    # Load the features list
    with open("model/model_features.json", 'r') as features_file:
        app.state.model_features = json.load(features_file)
    
    # Load demographics data
    app.state.demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={'zipcode': str})
    
    print("Model and demographic data loaded successfully")
    
    yield
    
    # Clean up resources on shutdown
    print("Shutting down, cleaning up resources...")
    app.state.model = None
    app.state.model_features = None
    app.state.demographics = None

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

app = create_app()

class FutureUnseen(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class FutureUnseenCreateRequest(FutureUnseen):
    yr_renovated: Optional[int] = 0

#BONUS
class FutureUnseenRequired(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: str  # Not in features list directly but needed to join with demographic data

class PredictionResponse(BaseModel):
    predicted_price: float
    property_details: FutureUnseen



#the BONUS is satisfied through the pydantic input validation
@app.post("/predict-prices")
async def predict_housing_prices(req: FutureUnseenRequired | FutureUnseenCreateRequest, request: Request):
    # Convert request to dictionary
    input_data = req.dict()
    
    # Extract features needed for the model
    input_df = pd.DataFrame([input_data])

    user_zipcode = input_data['zipcode']
    user_demographics = request.app.state.demographics[request.app.state.demographics['zipcode'] == user_zipcode]
    
    if user_demographics.empty:
        # Handle missing zipcode
        raise HTTPException(status_code=400, detail=f"No demographic data for zipcode {user_zipcode}")
    
    input_df = pd.merge(input_df, user_demographics, on="zipcode", how="inner")
    print(input_df.shape)
    # Keep only the columns needed by the model and in the correct order
    model_input = input_df[request.app.state.model_features]
    
    # Make prediction
    prediction = request.app.state.model.predict(model_input)[0]


    return {
        "predicted_price": float(prediction),
        "property_details": req
    }


@app.post("/predict-prices-required")
async def predict_housing_prices_required(req: FutureUnseenRequired, request: Request):
    # Convert request to dictionary
    input_data = req.dict()
    
    # Extract features needed for the model
    input_df = pd.DataFrame([input_data])

    user_zipcode = input_data['zipcode']
    user_demographics = request.app.state.demographics[request.app.state.demographics['zipcode'] == user_zipcode]
    
    if user_demographics.empty:
        # Handle missing zipcode
        raise HTTPException(status_code=400, detail=f"No demographic data for zipcode {user_zipcode}")
    
    input_df = pd.merge(input_df, user_demographics, on="zipcode", how="inner")
    print(input_df.shape)
    # Keep only the columns needed by the model and in the correct order
    model_input = input_df[request.app.state.model_features]
    
    # Make prediction
    prediction = request.app.state.model.predict(model_input)[0]


    return {
        "predicted_price": float(prediction),
        "property_details": req
    }
