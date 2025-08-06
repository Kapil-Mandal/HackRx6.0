# app/main.py
import os
import requests
import tempfile
import json
from fastapi import FastAPI, Header, HTTPException
from typing import List, Dict

# Import your models and services
from .models import QueryRequest, QueryResponse
from . import services
from . import utils

# --- FastAPI App ---
app = FastAPI(title="Intelligent Query-Retrieval System")

EXPECTED_BEARER_TOKEN = "5f5247a861f9c5c82791ab8ebdded7644021b6ff466ee2ba5b6a61328f83c9a0"


def process_and_store_document(filepath: str):
    """Helper function to run the document processing pipeline."""
    print(f"Starting processing for {filepath}...")
    text = utils.get_text_from_pdf(filepath)
    if not text:
        print("Failed to extract text from PDF.")
        raise ValueError("Failed to extract text from the provided PDF.")

    chunks = utils.get_text_chunks(text)
    vector_store = services.create_vector_store(chunks)
    services.save_vector_store(vector_store)
    print(f"Successfully processed and stored vector index for {filepath}")
    os.remove(filepath)  # Clean up the downloaded file


@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Intelligent Query-Retrieval System!"}


@app.post("/hackrx/run", response_model=QueryResponse)
async def run_submission(
        request: QueryRequest,
        authorization: str = Header(None)
):
    # --- Authentication ---
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    if token != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # --- Download the file from the URL ---
        print(f"Downloading document from: {request.documents}")
        response = requests.get(request.documents)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the downloaded content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_filepath = temp_file.name
        print(f"Document downloaded to temporary file: {temp_filepath}")

        # --- Process Document and then Query ---
        process_and_store_document(temp_filepath)

        # --- Query Processing ---
        final_answers = []
        for q in request.questions:
            result = services.process_query(q)
            final_answers.append({
                "question": q,
                "answer": result.get("answer", "No answer found."),
                "rationale": result.get("source_clauses", [])
            })

        return {"answers": final_answers}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        # Catch any other exception and return a detailed 500 error
        print(f"!!! A critical error occurred in the main endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")