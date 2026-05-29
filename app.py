import os
import shutil
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "results"       not in st.session_state: st.session_state.results       = None
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "pipeline_done" not in st.session_state: st.session_state.pipeline_done = False
if "processing"    not in st.session_state: st.session_state.processing    = False

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 AI Video Assistant")
    st.divider()

    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="https://youtu.be/...",
    )

    language = st.selectbox(
        "Video Language",
        options=["en", "hi"],
        format_func=lambda x: "English" if x == "en" else "Hindi / Hinglish",
    )

    analyze_btn = st.button(
        "🚀 Analyze Video",
        disabled=st.session_state.processing,
        use_container_width=True,
    )

    st.divider()
    st.caption("Built with LangChain · Groq · ChromaDB")


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_pipeline(source: str, language: str):
    from utils.audio_processor import process_input
    from core.transcriber import transcribe_all
    from core.translator import translate_transcript
    from core.summarizer import summarize, generate_title
    from core.extractor import extract_action_items, extract_key_decisions, extract_questions
    from core.vector_store import build_vector_store
    from core.rag_engine import load_rag_chain

    st.session_state.processing = True

    try:
        if os.path.exists("transcript.txt"):
            os.remove("transcript.txt")
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        with st.status("⏳ Running pipeline...", expanded=True) as status:

            st.write("🎙️ Step 1: Downloading & transcribing audio...")
            chunks     = process_input(source)
            transcript = transcribe_all(chunks, language=language)
            with open("transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)

            st.write("🌐 Step 2: Detecting language & translating...")
            clean_transcript = translate_transcript(transcript)

            st.write("📝 Step 3: Generating title & summary...")
            title   = generate_title(clean_transcript)
            summary = summarize(clean_transcript)

            st.write("🔍 Step 4: Extracting action items, decisions & questions...")
            action_items   = extract_action_items(clean_transcript)
            key_decisions  = extract_key_decisions(clean_transcript)
            open_questions = extract_questions(clean_transcript)

            st.write("🗄️ Step 5: Building vector store...")
            build_vector_store(clean_transcript)

            st.write("🤖 Step 6: Loading RAG chain...")
            rag_chain = load_rag_chain()

            status.update(label="✅ Pipeline complete!", state="complete")

        st.session_state.results = {
            "title":          title,
            "transcript":     clean_transcript,
            "summary":        summary,
            "action_items":   action_items,
            "key_decisions":  key_decisions,
            "open_questions": open_questions,
            "rag_chain":      rag_chain,
        }
        st.session_state.pipeline_done = True
        st.session_state.chat_history  = []

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        raise e

    finally:
        st.session_state.processing = False
        st.rerun()


# ─────────────────────────────────────────────
# Trigger
# ─────────────────────────────────────────────

if analyze_btn:
    if not source.strip():
        st.warning("⚠️ Please enter a YouTube URL or file path.")
    else:
        run_pipeline(source, language)


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────

if not st.session_state.pipeline_done:
    st.title("🎬 AI Video Assistant")
    st.markdown("Enter a YouTube URL in the sidebar and click **Analyze Video** to get started.")
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📝", "Summary")
    col2.metric("✅", "Actions")
    col3.metric("🔑", "Decisions")
    col4.metric("❓", "Questions")
    col5.metric("💬", "RAG Chat")

else:
    results = st.session_state.results

    st.title(f"🎬 {results['title']}")
    st.divider()

    tab_summary, tab_actions, tab_transcript, tab_chat = st.tabs([
        "📝 Summary",
        "✅ Actions & Decisions",
        "📄 Transcript",
        "💬 Chat with Video",
    ])

    # ── Tab 1: Summary ────────────────────────
    with tab_summary:
        st.markdown(results["summary"])

    # ── Tab 2: Actions & Decisions ────────────
    with tab_actions:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Action Items")
            st.markdown(results["action_items"])
            st.divider()
            st.subheader("🔑 Key Decisions")
            st.markdown(results["key_decisions"])
        with col2:
            st.subheader("❓ Open Questions")
            st.markdown(results["open_questions"])

    # ── Tab 3: Transcript ─────────────────────
    with tab_transcript:
        st.subheader("📄 Full Transcript")
        st.text_area("", value=results["transcript"], height=500, label_visibility="collapsed")

    # ── Tab 4: Chat ───────────────────────────
    with tab_chat:
        st.subheader("💬 Chat with Video")
        st.caption("Powered by Groq LLaMA 3.3 70B + Reciprocal Rank Fusion")

        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about the video..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    from core.rag_engine import ask_question
                    answer = ask_question(results["rag_chain"], prompt)
                st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
