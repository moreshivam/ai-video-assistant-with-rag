import os
import yt_dlp
from pydub import AudioSegment

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Directory where downloaded and processed audio files will be saved
DOWNLOAD_DIR = "downloads"


# ─────────────────────────────────────────────
# Function 1: Download Audio from YouTube
# ─────────────────────────────────────────────

def download_youtube_audio(url: str) -> str:
    """Downloads audio from a YouTube URL and saves it as a WAV file."""

    # --- Step 1: Ensure download directory exists ---
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # --- Step 2: Configure yt_dlp download options ---
    ydl_opts = {
        "format": "bestaudio/best",                        # pick highest quality audio stream
        "outtmpl": f"{DOWNLOAD_DIR}/%(title)s.%(ext)s",   # save with video title as filename
        "nocheckcertificate": True,        # bypass SSL issues on corporate/restricted networks
        "ffmpeg_location": r"C:\Users\Hp\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",       # convert to WAV after download
            "preferredquality": "192",     # 192kbps audio quality
        }],
    }

    # --- Step 3: Download and extract audio ---
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # --- Step 4: Build the final .wav filename ---
        # yt_dlp gives the original extension filename, we swap it to .wav
        filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".wav"

    return filename


# ─────────────────────────────────────────────
# Function 2: Convert Local File to WAV
# ─────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """Converts any local audio/video file to mono 16kHz WAV (Whisper-compatible format)."""

    # --- Step 1: Load the audio/video file ---
    audio = AudioSegment.from_file(input_path)

    # --- Step 2: Normalize audio for Whisper ---
    # Whisper works best with mono audio at 16kHz sample rate
    audio = audio.set_channels(1).set_frame_rate(16000)

    # --- Step 3: Export as WAV ---
    wav_path = input_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    return wav_path


# ─────────────────────────────────────────────
# Function 3: Split Audio into Chunks
# ─────────────────────────────────────────────

def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list[str]:
    """Splits a WAV file into smaller chunks to avoid Whisper memory limits."""

    # --- Step 1: Load the WAV file ---
    audio = AudioSegment.from_wav(wav_path)

    # --- Step 2: Calculate chunk size in milliseconds ---
    # 5 min default keeps chunks under Groq's 25MB API limit
    chunk_ms = chunk_minutes * 60 * 1000  # e.g. 5 min * 60 sec * 1000 ms = 300,000 ms

    # --- Step 3: Slice audio and save each chunk ---
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = wav_path.rsplit(".", 1)[0] + f"_chunk{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    # --- Step 4: Return list of chunk file paths ---
    return chunks


# ─────────────────────────────────────────────
# Function 4: Main Entry Point
# ─────────────────────────────────────────────

def process_input(source: str) -> list[str]:
    """
    Main entry point. Accepts a YouTube URL or a local file path.
    Returns a list of WAV chunk file paths ready for transcription.
    """

    # --- Step 1: Detect input type (URL or local file) ---
    if source.startswith("http://") or source.startswith("https://"):

        # --- Step 2a: Download audio from YouTube ---
        wav_path = download_youtube_audio(source)

        # --- Step 2b: Normalize to mono 16kHz after download ---
        # yt_dlp downloads at original quality (e.g. 48kHz stereo) which exceeds
        # Groq's 25MB API limit — normalize before chunking
        wav_path = convert_to_wav(wav_path)
    else:

        # --- Step 2c: Convert local file to normalized WAV ---
        wav_path = convert_to_wav(source)

    # --- Step 3: Split audio into chunks and return ---
    return chunk_audio(wav_path)
