# DI Interview: Product classifier 

# Product Classifier Model Deployment

# Overview

This project involves developing a product classification model using text data. The pipeline includes data extraction, feature engineering, model training, evaluation, and API deployment. The best-performing model is integrated into a FastAPI-based application.

# Project Structure

Project Structure
app/

__init__.py
main.py: FastAPI app to handle product classification requests
__pycache__/
training/

baseline_models/: Contains baseline models
mbert_models/: Best-performing model (mBERT trained on product titles)
xlmr_models/: Other transformer-based models
load_data_from_BQ.py: Script to fetch dataset from BigQuery
.gitignore

README.md

requirements.txt: List of dependencies

# Data Processing

# Data Extraction:

load_data_from_BQ.py fetches raw product data from BigQuery.

# Feature Engineering & Cleaning:

# Data preprocessing is performed in data_analysis.ipynb, which includes:

Text cleaning

Feature engineering

Train-test-validation splitting (ensuring consistency across models)

Model Training & Evaluation

Baseline Models: Simple models used as a benchmark.

# Transformer Models:

mbert_models/: Trained using product title text (selected as the best model).

xlmr_models/: Alternative transformer-based models.

All models are trained and evaluated consistently using the same split of train-test-validation data.

The best-performing model (mBERT) is selected and used in deployment.

# API Deployment

# The FastAPI application (app/main.py) provides a POST endpoint for product classification:

POST /predict
{
    "product_title": "Sample Product Title"
}

The API returns the predicted product class based on the provided title.

Postman is used to test the API (see attached Postman test results screenshot).

# Testing the API

The API was tested using Postman, and the result successfully shows the predicted product class based on title input.

# Requirements & Installation

# Install dependencies using:

pip install -r requirements.txt

# Run the FastAPI application:

uvicorn app.main:app --reload

# Notes

The project ensures modular development by keeping data processing, model training, and API deployment separate.

The best-performing model (mBERT trained on product titles) is deployed via FastAPI for real-time classification.

![Picture1](https://github.com/user-attachments/assets/44488d25-abde-4438-81de-d75c31720e79)


