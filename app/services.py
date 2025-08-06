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


# --- Query Processing ---

def get_conversational_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Creates the conversational retrieval chain."""
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    # The prompt is now simpler as we only need the direct answer string.
    custom_prompt_template = """
    Use the following pieces of context to answer the user's question.
    If you don't know the answer based on the context, just say that you don't know. Do not make up an answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False  # We don't need the source docs anymore
    )
    return chain


def process_query_with_store(query: str, vector_store: FAISS, chat_history: List = []) -> str:
    """
    Processes a single query against a provided in-memory vector store.
    Returns only the answer string.
    """
    try:
        chain = get_conversational_chain(vector_store)
        print(f"Processing query: {query}")
        result = chain({"question": query, "chat_history": chat_history})
        answer = result.get('answer', "Could not find an answer.")
        return answer
    except Exception as e:
        print(f"!!! An error occurred during query processing: {e}")
        return f"An error occurred: {str(e)}"