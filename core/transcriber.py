import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Groq's hosted Whisper model — supports English, Hindi, Hinglish and 90+ languages
WHISPER_MODEL = "whisper-large-v3"

# Disable SSL verification to handle corporate/restricted network certificate issues
_http_client = httpx.Client(verify=False)

# Initialize Groq client using API key from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"), http_client=_http_client)


# ─────────────────────────────────────────────
# Function 1: Transcribe a Single Audio Chunk
# ─────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, language: str = "en") -> str:
    """
    Sends a single audio chunk to Groq Whisper API and returns the transcript.
    Supports 'en' (English) and 'hi' (Hindi/Hinglish).
    """

    # --- Step 1: Open the audio chunk file ---
    with open(chunk_path, "rb") as audio_file:

        # --- Step 2: Send to Groq Whisper API ---
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            language=language,        # language hint improves accuracy
            response_format="text",   # return plain text, not JSON
        )

    # --- Step 3: Return the transcript text ---
    return response


# ─────────────────────────────────────────────
# Function 2: Transcribe All Chunks
# ─────────────────────────────────────────────

def transcribe_all(chunk_paths: list[str], language: str = "en") -> str:
    """
    Transcribes all audio chunks sequentially and concatenates into one full transcript.
    chunk_paths: list of .wav file paths (output from audio_processor.chunk_audio)
    """

    # --- Step 1: Transcribe each chunk one by one ---
    transcripts = []
    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

        # --- Step 2: Get transcript for this chunk ---
        text = transcribe_chunk(chunk_path, language=language)
        transcripts.append(text)

        # --- Step 3: Clean up chunk file after transcription ---
        os.remove(chunk_path)

    # --- Step 4: Join all chunk transcripts into one string ---
    full_transcript = " ".join(transcripts)

    return full_transcript
