# app/services.py
import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any

# Load API key from environment variables
from dotenv import load_dotenv

load_dotenv()


# --- Vector Store Management ---

def create_vector_store(chunks: List[str]) -> FAISS:
    """Creates a FAISS vector store from document chunks."""
    try:
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        print("Embeddings created successfully.")

        print("Creating vector store...")
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"!!! An error occurred during vector store creation: {e}")
        raise


def save_vector_store(vector_store: FAISS, path: str = "vector_store/faiss_index"):
    """Saves the vector store to a local path."""
    vector_store.save_local(path)


def load_vector_store(path: str = "vector_store/faiss_index") -> FAISS:
    """Loads a vector store from a local path."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


# --- Query Processing ---

def get_conversational_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Creates the conversational retrieval chain."""
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    custom_prompt_template = """
    Use the following pieces of context to answer the user's question.
    If you don't know the answer based on the context, just say that you don't know. Do not make up an answer.
    Your response must be a single JSON object with two keys: "answer" and "source_clauses".
    The "source_clauses" should be a list of the exact text segments from the context that you used.

    Context: {context}
    Question: {question}

    JSON Response:
    """
    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return chain


def process_query(query: str, chat_history: List = []) -> Dict[str, Any]:
    """Processes a single query against the loaded vector store."""
    try:
        vector_store = load_vector_store()
        chain = get_conversational_chain(vector_store)
        print(f"Processing query: {query}")
        result = chain({"question": query, "chat_history": chat_history})
        answer = result['answer']

        try:
            response_data = json.loads(answer)
        except json.JSONDecodeError:
            source_documents = [doc.page_content for doc in result['source_documents']]
            response_data = {"answer": answer, "source_clauses": source_documents}
        return response_data
    except FileNotFoundError:
        return {"answer": "Error: Vector store not found. Please process the document first.", "source_clauses": []}
    except Exception as e:
        print(f"!!! An error occurred during query processing: {e}")
        # Re-raise the exception to be caught by the main endpoint
        raise
