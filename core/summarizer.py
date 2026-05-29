import os
from dotenv import load_dotenv
import core.ssl_fix  # disables SSL verification for corporate/restricted networks
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MISTRAL_MODEL = "mistral-small-latest"


# ─────────────────────────────────────────────
# Function 1: Initialize LLM
# ─────────────────────────────────────────────

def get_llm() -> ChatMistralAI:
    """
    Returns a ChatMistralAI instance.
    Temperature 0.3 keeps summaries consistent and factual (less creative).
    """
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


# ─────────────────────────────────────────────
# Function 2: Split Transcript into Chunks
# ─────────────────────────────────────────────

def split_transcript(transcript: str) -> list[str]:
    """
    Splits a long transcript into overlapping chunks for map-reduce summarization.
    Uses 200-char overlap so context isn't lost at chunk boundaries.
    """

    # --- Step 1: Configure the text splitter ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,      # max characters per chunk
        chunk_overlap=200,    # overlap to preserve context between chunks
    )

    # --- Step 2: Split and return as list of strings ---
    chunks = splitter.split_text(transcript)
    print(f"Transcript split into {len(chunks)} chunk(s) for summarization.")
    return chunks


# ─────────────────────────────────────────────
# Function 3: Summarize Full Transcript
# ─────────────────────────────────────────────

def summarize(transcript: str) -> str:
    """
    Summarizes a full transcript using a map-reduce approach:
    1. MAP:    Summarize each chunk individually
    2. REDUCE: Combine all partial summaries into one final structured summary
    """

    llm = get_llm()
    parser = StrOutputParser()

    # --- Step 1: Split transcript into chunks ---
    chunks = split_transcript(transcript)

    # --- Step 2: MAP — summarize each chunk individually ---
    chunk_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Summarize the given portion of a video transcript concisely, preserving all key points."),
        ("human", "Summarize this portion of a video transcript:\n\n{chunk}")
    ])
    chunk_chain = chunk_summary_prompt | llm | parser

    print("Summarizing individual chunks...")
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = chunk_chain.invoke({"chunk": chunk})
        partial_summaries.append(summary)

    # --- Step 3: REDUCE — combine partial summaries into final summary ---
    combined = "\n\n".join(partial_summaries)

    final_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional content summarizer.
Combine the following partial summaries into one final, well-structured summary.

Format your response as:
**Overview:** (2-3 sentence overview of the video)

**Key Topics Covered:**
- (bullet points)

**Main Points:**
- (bullet points with the most important insights)

**Conclusion:**
(1-2 sentence conclusion)
"""),
        ("human", "Combine these partial summaries into a final structured summary:\n\n{summaries}")
    ])
    final_chain = final_summary_prompt | llm | parser

    print("Generating final structured summary...")
    final_summary = final_chain.invoke({"summaries": combined})

    return final_summary


# ─────────────────────────────────────────────
# Function 4: Generate Video Title
# ─────────────────────────────────────────────

def generate_title(transcript: str) -> str:
    """
    Generates a short, professional title for the video based on
    the first 2000 characters of the transcript.
    """

    llm = get_llm()
    parser = StrOutputParser()

    # --- Step 1: Use only the beginning of the transcript ---
    # First 2000 chars is enough context to generate a good title
    sample = transcript[:2000]

    # --- Step 2: Build and invoke the title prompt ---
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a short professional video title (max 8 words). Return only the title, no quotes or extra text."),
        ("human", "{transcript}")
    ])
    chain = title_prompt | llm | parser

    # --- Step 3: Return the generated title ---
    title = chain.invoke({"transcript": sample})
    return title.strip()
