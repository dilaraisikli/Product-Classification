from fastapi import FastAPI
from pydantic import BaseModel
from app import predict_product_type 

# Initialize FastAPI app
app = FastAPI()

# Request body structure
class ProductRequest(BaseModel):
    title: str

# API endpoint
@app.post("/classify")
async def classify_product(request: ProductRequest):
    return predict_product_type(request.title)
