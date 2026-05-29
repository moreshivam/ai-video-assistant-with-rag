import os
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.translator import translate_transcript
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question

# ─────────────────────────────────────────────
# Test: Full Pipeline
# ─────────────────────────────────────────────

VIDEO_URL    = "https://youtu.be/_pEEJu-2KKM?si=SuXsyyDw3h21_R2d"
TRANSCRIPT_FILE = "transcript.txt"
CHROMA_DB_DIR   = "chroma_db"

# ── Step 1: Transcription ─────────────────────
if os.path.exists(TRANSCRIPT_FILE):
    print("Step 1: Found existing transcript.txt — skipping download & transcription.")
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        transcript = f.read()
else:
    print("Step 1: Downloading and processing audio...")
    chunks = process_input(VIDEO_URL)
    transcript = transcribe_all(chunks, language="en")
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Transcript saved to: {TRANSCRIPT_FILE}")

print(f"Transcript length: {len(transcript)} characters")

# ── Step 2: Translation ───────────────────────
print("\nStep 2: Detecting language & translating if needed...")
clean_transcript = translate_transcript(transcript)

# ── Step 3: Title ─────────────────────────────
print("\nStep 3: Generating video title...")
title = generate_title(clean_transcript)
print(f"\n── TITLE ───────────────────────────────────")
print(title)
print("────────────────────────────────────────────")

# ── Step 4: Summary ───────────────────────────
print("\nStep 4: Summarizing transcript...")
summary = summarize(clean_transcript)
print(f"\n── SUMMARY ─────────────────────────────────")
print(summary)
print("────────────────────────────────────────────")

# ── Step 5: Extraction ────────────────────────
print("\nStep 5: Extracting action items, decisions & questions...")
action_items  = extract_action_items(clean_transcript)
key_decisions = extract_key_decisions(clean_transcript)
questions     = extract_questions(clean_transcript)

print(f"\n── ACTION ITEMS ────────────────────────────")
print(action_items)
print(f"\n── KEY DECISIONS ───────────────────────────")
print(key_decisions)
print(f"\n── OPEN QUESTIONS ──────────────────────────")
print(questions)
print("────────────────────────────────────────────")

# ── Step 6: RAG Chain ─────────────────────────
print("\nStep 6: Building RAG chain with Reciprocal Rank Fusion...")
if os.path.exists(CHROMA_DB_DIR):
    print("Found existing ChromaDB — loading from disk.")
    rag_chain = load_rag_chain()
else:
    rag_chain = build_rag_chain(clean_transcript)

# ── Step 7: Ask questions via RAG ────────────
print("\nStep 7: Testing RAG Q&A...")
test_questions = [
    "What is document intelligence?",
    "What are AI agents and how do they work?",
    "What is RAG fusion and how does RRF work?",
]
for q in test_questions:
    ask_question(rag_chain, q)
    print()

# ── Save all results ──────────────────────────
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(f"TITLE: {title}\n\n{summary}\n\n")
    f.write("── ACTION ITEMS ────────────────────────────\n")
    f.write(action_items + "\n\n")
    f.write("── KEY DECISIONS ───────────────────────────\n")
    f.write(key_decisions + "\n\n")
    f.write("── OPEN QUESTIONS ──────────────────────────\n")
    f.write(questions)

print("\nAll results saved to: summary.txt")
