import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Lightweight HuggingFace embedding model — fast, no API key needed, runs locally
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Directory where ChromaDB will persist the vector store on disk
CHROMA_DB_DIR = "chroma_db"

# ChromaDB collection name
COLLECTION_NAME = "video_transcript"


# ─────────────────────────────────────────────
# Function 1: Get Embedding Model
# ─────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFace embedding model instance.
    all-MiniLM-L6-v2 is a lightweight 384-dim model — fast and good for semantic search.
    Runs locally, no API key needed.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use CPU (change to "cuda" if GPU available)
        encode_kwargs={"normalize_embeddings": True},  # normalize for cosine similarity
    )


# ─────────────────────────────────────────────
# Function 2: Split Transcript into Chunks
# ─────────────────────────────────────────────

def split_transcript(transcript: str) -> list[str]:
    """
    Splits the transcript into smaller chunks for embedding.
    Smaller chunks (500 chars) give more precise retrieval results in RAG.
    """

    # --- Step 1: Configure text splitter ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # smaller than summarizer — better retrieval precision
        chunk_overlap=50,    # small overlap to preserve context at boundaries
    )

    # --- Step 2: Split and return chunks ---
    chunks = splitter.split_text(transcript)
    print(f"Transcript split into {len(chunks)} chunks for vector store.")
    return chunks


# ─────────────────────────────────────────────
# Function 3: Build Vector Store
# ─────────────────────────────────────────────

def build_vector_store(transcript: str) -> Chroma:
    """
    Splits the transcript, generates HuggingFace embeddings,
    and stores them in a persistent ChromaDB vector store.
    Returns the Chroma vectorstore instance.
    """

    # --- Step 1: Split transcript into chunks ---
    chunks = split_transcript(transcript)

    # --- Step 2: Load embedding model ---
    print("Loading HuggingFace embedding model...")
    embeddings = get_embeddings()

    # --- Step 3: Create ChromaDB vector store and embed all chunks ---
    print("Embedding chunks and storing in ChromaDB...")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,  # saves to disk so we don't re-embed every time
    )

    print(f"Vector store saved to: {CHROMA_DB_DIR}/")
    return vector_store


# ─────────────────────────────────────────────
# Function 4: Load Existing Vector Store
# ─────────────────────────────────────────────

def load_vector_store() -> Chroma:
    """
    Loads an existing ChromaDB vector store from disk.
    Use this instead of build_vector_store() if embeddings already exist.
    """

    # --- Step 1: Load embedding model (must match what was used to build) ---
    embeddings = get_embeddings()

    # --- Step 2: Load from disk ---
    print(f"Loading vector store from: {CHROMA_DB_DIR}/")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    return vector_store


# ─────────────────────────────────────────────
# Function 5: Get Retriever
# ─────────────────────────────────────────────

def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Returns a retriever from the vector store.
    k = number of most relevant chunks to retrieve per query.
    """

    # --- Return retriever with top-k similarity search ---
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
