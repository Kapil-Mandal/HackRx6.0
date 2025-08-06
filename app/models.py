# app/models.py
from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    """
    Defines the structure for the incoming JSON request to the /hackrx/run endpoint.
    """
    documents: str = Field(..., description="A public URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to ask about the document.")

class AnswerResponse(BaseModel):
    """
    Defines the structure for the final JSON response, which must match
    the hackathon's expected format: a simple list of answer strings.
    """
    answers: List[str]