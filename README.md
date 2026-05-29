# 🎬 AI Video Assistant with RAG

An end-to-end AI system that takes any YouTube video or local video file, processes it through an intelligent pipeline, and lets you **chat with the video content** using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🎙️ **Transcription** — Groq Whisper converts audio to text (cloud-based, no local model)
- 🌐 **Auto Translation** — Detects Hindi/Hinglish and translates to clean English
- 📝 **Smart Summary** — Structured summary with key topics, main points & conclusion
- ✅ **Action Items** — Extracts tasks, owners and deadlines from the video
- 🔑 **Key Decisions** — Identifies important decisions made
- ❓ **Open Questions** — Surfaces unresolved questions and follow-ups
- 💬 **RAG Chat** — Ask anything about the video using Reciprocal Rank Fusion RAG
- 🖥️ **Streamlit UI** — Clean web interface with tabs for each feature

---

## 🏗️ Architecture

```
YouTube URL / Local File
        ↓
audio_processor.py   → download, normalize to 16kHz mono, chunk audio
        ↓
transcriber.py       → Groq Whisper → raw transcript
        ↓
translator.py        → auto-detect language → translate if Hinglish
        ↓
summarizer.py        → Map-Reduce → structured summary + title
        ↓
extractor.py         → Map-Reduce → action items, decisions, questions
        ↓
vector_store.py      → HuggingFace embeddings → ChromaDB vector store
        ↓
rag_engine.py        → RRF (4 query variants) → Groq LLaMA → answers
        ↓
main.py / app.py     → CLI or Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Speech-to-Text | Groq Whisper `whisper-large-v3` |
| Translation & Cleaning | Mistral AI `mistral-small-latest` |
| Summarization | Mistral AI + LangChain LCEL |
| Extraction | Mistral AI + LangChain LCEL |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (persistent on disk) |
| RAG LLM | Groq `llama-3.3-70b-versatile` |
| RAG Strategy | Reciprocal Rank Fusion (RRF) |
| Frontend | Streamlit |
| Audio Processing | yt-dlp, pydub, FFmpeg |

---

## 📁 Project Structure

```
ai-video-assistant-with-rag/
├── core/
│   ├── transcriber.py     ← Groq Whisper speech-to-text
│   ├── translator.py      ← Hinglish → English via Mistral
│   ├── summarizer.py      ← Map-Reduce summarization
│   ├── extractor.py       ← Extract actions, decisions, questions
│   ├── vector_store.py    ← ChromaDB + HuggingFace embeddings
│   ├── rag_engine.py      ← Reciprocal Rank Fusion RAG
│   └── ssl_fix.py         ← SSL patch for restricted networks
├── utils/
│   └── audio_processor.py ← Download & process audio
├── main.py                ← CLI pipeline orchestrator
├── app.py                 ← Streamlit web UI
├── test.py                ← Pipeline test script
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/moreshivam/ai-video-assistant-with-rag.git
cd ai-video-assistant-with-rag
```

### 2. Create virtual environment
```bash
# Using uv (recommended)
uv venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
uv pip install -r requirements.txt --system-certs
```

### 3. Install FFmpeg
```bash
# Windows
winget install --id Gyan.FFmpeg --source winget

# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### 4. Configure API keys
Create a `.env` file in the project root:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API keys:
- **Groq** → https://console.groq.com
- **Mistral** → https://console.mistral.ai

---

## ▶️ Usage

### Option 1: CLI (Terminal)
```bash
python main.py
```
```
Enter YouTube URL or local file path: https://youtu.be/...
Language (en/hi) [default: en]: en
```
Runs the full pipeline, prints results, and opens an interactive chat session.

### Option 2: Streamlit UI
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 💬 RAG Strategy — Reciprocal Rank Fusion

Instead of basic similarity search, we use **Reciprocal Rank Fusion (RRF)**:

1. Generate **4 alternative phrasings** of the user's question
2. Retrieve top chunks for **each variant** independently from ChromaDB
3. Re-rank using RRF formula: `score(doc) = Σ 1 / (k + rank)`
4. Documents appearing consistently at the top across all queries score highest
5. Top 5 fused chunks are passed as context to Groq LLaMA for the final answer

This significantly improves retrieval accuracy over basic similarity search.

---

## 🔑 API Keys Required

| Service | Used For | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com) | Whisper STT + LLaMA RAG | ✅ Yes |
| [Mistral AI](https://console.mistral.ai) | Translation, Summary, Extraction | ✅ Yes |
| HuggingFace | Embeddings (runs locally) | ✅ No key needed |

---

## 📌 Notes

- On first run, the HuggingFace embedding model (~90MB) will be downloaded and cached automatically
- `transcript.txt` and `chroma_db/` are saved locally to avoid re-processing on subsequent runs
- Works with English and Hindi/Hinglish videos
- Tested on Python 3.10+
