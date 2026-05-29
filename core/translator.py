import os
from dotenv import load_dotenv
import core.ssl_fix  # disables SSL verification for corporate/restricted networks
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Mistral model used for translation/cleaning
MISTRAL_MODEL = "mistral-small-latest"

# Initialize Mistral LLM via LangChain
llm = ChatMistralAI(
    model=MISTRAL_MODEL,
    api_key=os.getenv("MISTRAL_API_KEY"),
)

# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

# Detects whether transcript is Hindi/Hinglish or English
DETECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a language detection assistant.
Analyze the given text and respond with ONLY one word:
- "hinglish" — if the text contains Hindi words, Devanagari script, or is a mix of Hindi and English
- "english"  — if the text is purely in English

Do not explain. Just reply with one word: hinglish or english."""),
    ("human", "{text}")
])

# Translates Hindi/Hinglish transcript to clean English
TRANSLATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional transcript translator.
Translate the following Hindi or Hinglish transcript into clean, readable English.

Rules:
- Translate all Hindi/Hinglish words and sentences to English
- Preserve the original meaning and all key information — do not summarize or remove content
- Remove filler words (uh, um, haan, toh, basically, you know)
- Return only the translated transcript, no explanations or extra text
"""),
    ("human", "Translate this transcript to English:\n\n{transcript}")
])


# ─────────────────────────────────────────────
# Function 1: Detect Language
# ─────────────────────────────────────────────

def detect_language(transcript: str) -> str:
    """
    Detects whether the transcript is 'english' or 'hinglish'.
    Uses a small sample (first 500 chars) to keep it fast and cheap.
    Returns: 'english' or 'hinglish'
    """

    # --- Step 1: Use only first 500 chars as a sample for detection ---
    sample = transcript[:500]

    # --- Step 2: Ask Mistral to detect the language ---
    chain = DETECTION_PROMPT | llm
    response = chain.invoke({"text": sample})

    # --- Step 3: Parse and return the result ---
    detected = response.content.strip().lower()
    print(f"Language detected: {detected}")

    # --- Step 4: Default to 'english' if response is unexpected ---
    return detected if detected in ("english", "hinglish") else "english"


# ─────────────────────────────────────────────
# Function 2: Translate a Single Chunk
# ─────────────────────────────────────────────

def translate_chunk(text: str) -> str:
    """Translates a single Hinglish chunk to clean English using Mistral."""

    # --- Step 1: Build and invoke the translation chain ---
    chain = TRANSLATION_PROMPT | llm
    response = chain.invoke({"transcript": text})

    # --- Step 2: Return translated text ---
    return response.content


# ─────────────────────────────────────────────
# Function 3: Translate Full Transcript
# ─────────────────────────────────────────────

def translate_transcript(transcript: str, chunk_size: int = 3000) -> str:
    """
    Main entry point. Detects language first — only translates if Hinglish.
    If already English, returns the transcript as-is (no API calls wasted).
    chunk_size: number of characters per chunk to stay within token limits.
    """

    # --- Step 1: Detect language of the transcript ---
    language = detect_language(transcript)

    # --- Step 2: Skip translation if already English ---
    if language == "english":
        print("Transcript is already in English — skipping translation.")
        return transcript

    # --- Step 3: Split Hinglish transcript into chunks ---
    # Mistral has token limits so we process in parts
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    print(f"Translating Hinglish transcript in {len(chunks)} chunk(s)...")

    # --- Step 4: Translate each chunk ---
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"  Translating chunk {i + 1}/{len(chunks)}...")
        translated = translate_chunk(chunk)
        translated_chunks.append(translated)

    # --- Step 5: Join all translated chunks into one clean transcript ---
    return " ".join(translated_chunks)
