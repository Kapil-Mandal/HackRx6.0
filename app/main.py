# app/main.py
import os
import requests
import tempfile
from fastapi import FastAPI, Header, HTTPException
from typing import List, Dict

# Import your models and services
from .models import QueryRequest, AnswerResponse
from . import services
from . import utils

# --- FastAPI App ---
app = FastAPI(title="Intelligent Query-Retrieval System")

EXPECTED_BEARER_TOKEN = "5f5247a861f9c5c82791ab8ebdded7644021b6ff466ee2ba5b6a61328f83c9a0"


@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Intelligent Query-Retrieval System!"}


@app.post("/hackrx/run", response_model=AnswerResponse)
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

    temp_filepath = None
    try:
        # --- Download the file from the URL ---
        print(f"Downloading document from: {request.documents}")
        response = requests.get(request.documents)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_filepath = temp_file.name
        print(f"Document downloaded to temporary file: {temp_filepath}")

        # --- Create Vector Store in Memory ---
        text = utils.get_text_from_pdf(temp_filepath)
        if not text:
            raise ValueError("Failed to extract text from the provided PDF.")

        chunks = utils.get_text_chunks(text)
        vector_store = services.create_vector_store(chunks)
        print("In-memory vector store created successfully.")

        # --- Process All Questions ---
        final_answers = []
        for q in request.questions:
            answer_string = services.process_query_with_store(q, vector_store)
            final_answers.append(answer_string)

        # --- Return response in the required format ---
        return {"answers": final_answers}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        print(f"!!! A critical error occurred in the main endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
    finally:
        # --- Clean up the temporary file ---
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Cleaned up temporary file: {temp_filepath}")
