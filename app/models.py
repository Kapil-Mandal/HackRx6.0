# app/models.py
from pydantic import BaseModel, Field
from typing import List, Dict

class QueryRequest(BaseModel):
    """
    Defines the structure for the incoming JSON request to the /hackrx/run endpoint.
    """
    documents: str = Field(..., description="A public URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class Answer(BaseModel):
    """
    Defines the structure for a single answer object.
    This is used for clear documentation but is part of the final dictionary in the response.
    """
    question: str
    answer: str
    rationale: List[str] = Field(..., description="Source clauses from the document used for the answer.")

class QueryResponse(BaseModel):
    """
    Defines the structure for the final JSON response from the /hackrx/run endpoint.
    """
    answers: List[Dict]