import os
from dotenv import load_dotenv
import core.ssl_fix  # disables SSL verification for corporate/restricted networks
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
RRF_K = 60          # RRF constant — higher = less aggressive rank difference
NUM_QUERIES = 3     # number of alternative queries to generate per question
TOP_N_DOCS = 5      # number of top-ranked docs to pass as context to LLM


# ─────────────────────────────────────────────
# Function 1: Initialize LLM
# ─────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """Groq LLaMA 3.3 70B — fast inference, used for both query generation and answering."""
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


# ─────────────────────────────────────────────
# Function 2: Generate Multiple Query Variants
# ─────────────────────────────────────────────

def generate_queries(question: str) -> list[str]:
    """
    Uses Groq to rephrase the user's question into NUM_QUERIES alternative versions.
    Different phrasings retrieve different relevant chunks — improving coverage.
    """

    llm = get_llm()
    parser = StrOutputParser()

    # --- Step 1: Prompt LLM to generate alternative questions ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"""Generate {NUM_QUERIES} different phrasings of the user's question to improve document retrieval.
Each phrasing should approach the question from a slightly different angle.
Return ONLY the questions, one per line, no numbering or extra text."""),
        ("human", "{question}")
    ])

    chain = prompt | llm | parser

    # --- Step 2: Get response and split into list ---
    response = chain.invoke({"question": question})
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]

    # --- Step 3: Always include the original question ---
    all_queries = [question] + queries
    print(f"Generated {len(all_queries)} query variants:")
    for q in all_queries:
        print(f"  → {q}")

    return all_queries


# ─────────────────────────────────────────────
# Function 3: Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(results: list[list], k: int = RRF_K) -> list:
    """
    Combines multiple ranked retrieval lists into one using RRF scoring.

    Formula: score(doc) = Σ 1 / (k + rank)
    - Documents appearing consistently at the top across multiple queries score highest
    - k=60 prevents top-ranked docs from dominating too aggressively

    results: list of retrieval result lists (one list per query variant)
    Returns: list of (document, score) tuples sorted by score descending
    """

    # --- Step 1: Accumulate RRF scores for each unique document ---
    rrf_scores = {}

    for result_list in results:
        for rank, doc in enumerate(result_list):

            # Use page content as unique key for the document
            doc_key = doc.page_content

            # --- RRF formula: add 1/(k + rank) to the doc's score ---
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {"score": 0.0, "doc": doc}
            rrf_scores[doc_key]["score"] += 1.0 / (k + rank + 1)

    # --- Step 2: Sort documents by score descending ---
    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in ranked]


# ─────────────────────────────────────────────
# Function 4: Format Documents for Prompt
# ─────────────────────────────────────────────

def format_docs(docs) -> str:
    """Joins retrieved document chunks into a single context string for the prompt."""
    return "\n\n".join([doc.page_content for doc in docs])


# ─────────────────────────────────────────────
# Function 5: RAG Prompt Template
# ─────────────────────────────────────────────

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Prompt that instructs LLM to answer ONLY from transcript context.
    Prevents hallucination by explicitly bounding the answer space.
    """
    return ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert video assistant. Answer the user's question
based ONLY on the video transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the video transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from video transcript:
{context}"""),
        ("human", "{question}"),
    ])


# ─────────────────────────────────────────────
# Function 6: RRF Retrieval Step
# ─────────────────────────────────────────────

def rrf_retrieve(question: str, retriever) -> str:
    """
    Full Reciprocal Rank Fusion retrieval:
    1. Generate multiple query variants
    2. Retrieve docs for each variant
    3. Re-rank using RRF
    4. Return top N docs as formatted context string
    """

    # --- Step 1: Generate query variants ---
    queries = generate_queries(question)

    # --- Step 2: Retrieve docs for each query variant ---
    all_results = []
    for query in queries:
        docs = retriever.invoke(query)
        all_results.append(docs)

    # --- Step 3: Re-rank using Reciprocal Rank Fusion ---
    fused_docs = reciprocal_rank_fusion(all_results)

    # --- Step 4: Take top N and format as context ---
    top_docs = fused_docs[:TOP_N_DOCS]
    print(f"RRF selected top {len(top_docs)} chunks as context.")

    return format_docs(top_docs)


# ─────────────────────────────────────────────
# Function 7: Build RAG Chain (first run)
# ─────────────────────────────────────────────

def build_rag_chain(transcript: str):
    """
    Builds the full RRF-RAG pipeline from scratch.
    Embeds the transcript into ChromaDB on first run.
    """

    # --- Step 1: Build vector store from transcript ---
    print("Building vector store from transcript...")
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()
    prompt = get_rag_prompt()

    # --- Step 2: Build LCEL chain with RRF retrieval ---
    rag_chain = (
        {
            # Use RRF retrieval instead of simple similarity search
            "context": RunnableLambda(lambda q: rrf_retrieve(q, retriever)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ─────────────────────────────────────────────
# Function 8: Load RAG Chain (subsequent runs)
# ─────────────────────────────────────────────

def load_rag_chain():
    """
    Loads RAG chain from existing ChromaDB on disk.
    Use after first run to skip re-embedding.
    """

    # --- Step 1: Load existing vector store ---
    print("Loading existing vector store from disk...")
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()
    prompt = get_rag_prompt()

    # --- Step 2: Build LCEL chain with RRF retrieval ---
    rag_chain = (
        {
            "context": RunnableLambda(lambda q: rrf_retrieve(q, retriever)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ─────────────────────────────────────────────
# Function 9: Ask a Question
# ─────────────────────────────────────────────

def ask_question(rag_chain, question: str) -> str:
    """Runs a question through the RRF-RAG chain and returns the answer."""

    print(f"\nQuestion: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
