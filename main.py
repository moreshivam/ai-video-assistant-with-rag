import os
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.translator import translate_transcript
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TRANSCRIPT_FILE = "transcript.txt"
CHROMA_DB_DIR   = "chroma_db"


# ─────────────────────────────────────────────
# Function 1: Run Full Pipeline
# ─────────────────────────────────────────────

def process_video(source: str, language: str = "en") -> dict:
    """
    Runs the full AI Video Assistant pipeline on a YouTube URL or local file.

    Steps:
    1. Download & process audio
    2. Transcribe (Groq Whisper)
    3. Translate if Hinglish (Mistral)
    4. Generate title (Mistral)
    5. Summarize (Mistral)
    6. Extract action items, decisions, questions (Mistral)
    7. Build RAG chain with RRF (ChromaDB + Groq)

    Returns a dict with all results + rag_chain for chat.
    """

    # ── Step 1: Transcription ─────────────────
    # Reuse existing transcript if available to avoid re-downloading
    if os.path.exists(TRANSCRIPT_FILE):
        print(">> Found existing transcript — skipping download & transcription.")
        with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        print(">> Step 1: Downloading and processing audio...")
        chunks = process_input(source)
        print(f"   Audio split into {len(chunks)} chunk(s)")

        print(">> Step 2: Transcribing with Groq Whisper...")
        transcript = transcribe_all(chunks, language=language)

        # Save transcript for reuse
        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"   Transcript saved to: {TRANSCRIPT_FILE}")

    print(f"   Transcript length: {len(transcript)} characters\n")

    # ── Step 2: Translation ───────────────────
    print(">> Step 3: Detecting language & translating if needed...")
    clean_transcript = translate_transcript(transcript)

    # ── Step 3: Generate Title ────────────────
    print("\n>> Step 4: Generating video title...")
    title = generate_title(clean_transcript)
    print(f"   Title: {title}")

    # ── Step 4: Summarize ─────────────────────
    print("\n>> Step 5: Summarizing transcript...")
    summary = summarize(clean_transcript)

    # ── Step 5: Extract ───────────────────────
    print("\n>> Step 6: Extracting action items, decisions & questions...")
    action_items  = extract_action_items(clean_transcript)
    key_decisions = extract_key_decisions(clean_transcript)
    open_questions = extract_questions(clean_transcript)

    # ── Step 6: Build RAG Chain ───────────────
    print("\n>> Step 7: Building RAG chain with Reciprocal Rank Fusion...")
    if os.path.exists(CHROMA_DB_DIR):
        print("   Found existing ChromaDB — loading from disk.")
        rag_chain = load_rag_chain()
    else:
        rag_chain = build_rag_chain(clean_transcript)

    print("\n>> Pipeline complete!\n")

    # ── Return all results ────────────────────
    return {
        "title":          title,
        "transcript":     clean_transcript,
        "summary":        summary,
        "action_items":   action_items,
        "key_decisions":  key_decisions,
        "open_questions": open_questions,
        "rag_chain":      rag_chain,
    }


# ─────────────────────────────────────────────
# Function 2: Display Results
# ─────────────────────────────────────────────

def display_results(results: dict):
    """Prints all pipeline results to the console in a formatted way."""

    print("\n" + "=" * 60)
    print(f"  VIDEO TITLE")
    print("=" * 60)
    print(results["title"])

    print("\n" + "=" * 60)
    print(f"  SUMMARY")
    print("=" * 60)
    print(results["summary"])

    print("\n" + "=" * 60)
    print(f"  ACTION ITEMS")
    print("=" * 60)
    print(results["action_items"])

    print("\n" + "=" * 60)
    print(f"  KEY DECISIONS")
    print("=" * 60)
    print(results["key_decisions"])

    print("\n" + "=" * 60)
    print(f"  OPEN QUESTIONS")
    print("=" * 60)
    print(results["open_questions"])


# ─────────────────────────────────────────────
# Function 3: Interactive Chat Loop
# ─────────────────────────────────────────────

def chat_loop(rag_chain):
    """
    Starts an interactive CLI chat session using the RAG chain.
    Type 'exit' or 'quit' to end the session.
    """

    print("\n" + "=" * 60)
    print("  CHAT WITH YOUR VIDEO  (type 'exit' to quit)")
    print("=" * 60)

    while True:

        # --- Get user input ---
        question = input("\nYou: ").strip()

        # --- Exit condition ---
        if question.lower() in ("exit", "quit", "q"):
            print("Ending chat session. Goodbye!")
            break

        # --- Skip empty input ---
        if not question:
            continue

        # --- Get answer from RAG chain ---
        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("   AI VIDEO ASSISTANT WITH RAG")
    print("=" * 60)

    # --- Get video source from user ---
    source = input("\nEnter YouTube URL or local file path: ").strip()

    if not source:
        print("No source provided. Exiting.")
        exit()

    # --- Ask for language ---
    lang = input("Language (en/hi) [default: en]: ").strip() or "en"

    # --- Run full pipeline ---
    results = process_video(source, language=lang)

    # --- Display all results ---
    display_results(results)

    # --- Save all results to file ---
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"TITLE\n{'='*60}\n{results['title']}\n\n")
        f.write(f"SUMMARY\n{'='*60}\n{results['summary']}\n\n")
        f.write(f"ACTION ITEMS\n{'='*60}\n{results['action_items']}\n\n")
        f.write(f"KEY DECISIONS\n{'='*60}\n{results['key_decisions']}\n\n")
        f.write(f"OPEN QUESTIONS\n{'='*60}\n{results['open_questions']}\n")
    print("\nAll results saved to: results.txt")

    # --- Start interactive chat ---
    chat_loop(results["rag_chain"])
