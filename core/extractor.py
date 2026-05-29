import os
from dotenv import load_dotenv
import core.ssl_fix  # disables SSL verification for corporate/restricted networks
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MISTRAL_MODEL = "mistral-small-latest"
CHUNK_SIZE = 3000   # characters per chunk
CHUNK_OVERLAP = 200 # overlap to avoid missing items at chunk boundaries


# ─────────────────────────────────────────────
# Function 1: Initialize LLM
# ─────────────────────────────────────────────

def get_llm() -> ChatMistralAI:
    """
    Returns a ChatMistralAI instance.
    Temperature 0.2 keeps extraction precise and consistent (very low creativity).
    """
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


# ─────────────────────────────────────────────
# Function 2: Split Transcript into Chunks
# ─────────────────────────────────────────────

def split_transcript(transcript: str) -> list[str]:
    """Splits a long transcript into overlapping chunks for extraction."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


# ─────────────────────────────────────────────
# Function 3: Build Extraction Chain
# ─────────────────────────────────────────────

def build_chain(system_prompt: str):
    """
    Builds a reusable LangChain LCEL chain for a given system prompt.
    Takes raw text as input and returns extracted content as string.
    """
    llm = get_llm()

    # --- Build chain: input → wrap in dict → prompt → LLM → parse to string ---
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


# ─────────────────────────────────────────────
# Function 4: Extract from Chunks (Map-Reduce)
# ─────────────────────────────────────────────

def extract_from_chunks(transcript: str, system_prompt: str, combine_prompt: str) -> str:
    """
    Shared map-reduce extractor used by all extraction functions.
    MAP:    Extract items from each chunk individually
    REDUCE: Combine and deduplicate all extracted items into a final list
    """

    # --- Step 1: Split transcript into chunks ---
    chunks = split_transcript(transcript)
    print(f"  Processing {len(chunks)} chunk(s)...")

    # --- Step 2: MAP — extract from each chunk ---
    chain = build_chain(system_prompt)
    partial_results = []
    for i, chunk in enumerate(chunks):
        print(f"  Extracting from chunk {i + 1}/{len(chunks)}...")
        result = chain.invoke(chunk)
        partial_results.append(result)

    # --- Step 3: REDUCE — combine all partial results into a clean final list ---
    combined = "\n\n".join(partial_results)
    combine_chain = build_chain(combine_prompt)
    final_result = combine_chain.invoke(combined)

    return final_result


# ─────────────────────────────────────────────
# Function 5: Extract Action Items
# ─────────────────────────────────────────────

def extract_action_items(transcript: str) -> str:
    """
    Extracts all action items from the transcript using chunked map-reduce.
    Each item includes: task description, owner, and deadline.
    """
    print("Extracting action items...")

    return extract_from_chunks(
        transcript,
        # MAP prompt — extract from each chunk
        system_prompt=(
            "You are an expert meeting analyst. From this transcript portion, "
            "extract all action items. For each provide:\n"
            "- Task description\n"
            "- Owner (who is responsible)\n"
            "- Deadline (if mentioned, else write 'Not specified')\n\n"
            "Format as a numbered list. If none found say 'No action items found.'"
        ),
        # REDUCE prompt — deduplicate and combine all chunks
        combine_prompt=(
            "Below are action items extracted from different parts of a transcript. "
            "Combine them into one clean, deduplicated numbered list. "
            "Remove duplicates. If none found say 'No action items found.'"
        )
    )


# ─────────────────────────────────────────────
# Function 6: Extract Key Decisions
# ─────────────────────────────────────────────

def extract_key_decisions(transcript: str) -> str:
    """
    Extracts all key decisions from the transcript using chunked map-reduce.
    """
    print("Extracting key decisions...")

    return extract_from_chunks(
        transcript,
        system_prompt=(
            "You are an expert meeting analyst. From this transcript portion, "
            "extract all key decisions made. Format as a numbered list. "
            "If none found say 'No key decisions found.'"
        ),
        combine_prompt=(
            "Below are key decisions extracted from different parts of a transcript. "
            "Combine them into one clean, deduplicated numbered list. "
            "Remove duplicates. If none found say 'No key decisions found.'"
        )
    )


# ─────────────────────────────────────────────
# Function 7: Extract Open Questions
# ─────────────────────────────────────────────

def extract_questions(transcript: str) -> str:
    """
    Extracts all unresolved questions or follow-up topics using chunked map-reduce.
    """
    print("Extracting open questions...")

    return extract_from_chunks(
        transcript,
        system_prompt=(
            "From this transcript portion, extract all unresolved questions "
            "or topics needing follow-up. Format as a numbered list. "
            "If none found say 'No open questions found.'"
        ),
        combine_prompt=(
            "Below are open questions extracted from different parts of a transcript. "
            "Combine them into one clean, deduplicated numbered list. "
            "Remove duplicates. If none found say 'No open questions found.'"
        )
    )
