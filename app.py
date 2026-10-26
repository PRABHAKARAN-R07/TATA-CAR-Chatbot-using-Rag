import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import load_rag, ask_tata_question

load_dotenv()

st.set_page_config(
    page_title="Tata Car Guide",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CARS = [
    "Punch", "Punch EV", "Nexon", "Nexon EV", "Curvv",
    "Curvv EV", "Harrier", "Harrier EV", "Safari", "Sierra",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --orange: #ff4328;
    --bg: #070b12;
    --panel: #0b111b;
    --card: #111a27;
    --border: #283447;
    --muted: #a9bdd4;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 0% 0%, #50130d 0%, transparent 27%),
        radial-gradient(ellipse at 100% 100%, #3c100b 0%, transparent 22%),
        linear-gradient(130deg, #080b11, #0b111b 65%, #060a11);
    color: #f4f7fc;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1600px;
    padding: 2rem 2.2rem 1rem;
}

h1, h2, h3, p, label {
    color: #f4f7fc;
}

.hero-title {
    font-size: clamp(36px, 4vw, 66px);
    line-height: 1.06;
    font-weight: 900;
    letter-spacing: -2px;
    margin-top: 45px;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #c0d0e2;
    font-size: clamp(18px, 2vw, 27px);
    font-weight: 600;
    margin-bottom: 24px;
}

.orange-line {
    height: 5px;
    width: 62px;
    border-radius: 8px;
    background: var(--orange);
    margin: 20px 0 35px;
    box-shadow: 0 0 15px #ff432866;
}

.feature {
    display: flex;
    gap: 18px;
    margin: 28px 0;
    align-items: flex-start;
}

.feature-icon {
    min-width: 60px;
    height: 60px;
    border: 1px solid var(--orange);
    border-radius: 14px;
    color: var(--orange);
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 28px;
    background: #10141c;
}

.feature-title {
    font-size: 17px;
    font-weight: 800;
    margin: 2px 0 5px;
}

.feature-description {
    color: #bfd0e2;
    font-size: 14px;
    line-height: 1.6;
}

.chat-panel {
    background: rgba(7, 12, 21, 0.90);
    border: 1px solid #2b3647;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 18px 55px #00000040;
}

.chat-header {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-bottom: 25px;
}

.chat-icon {
    border: 1px solid var(--orange);
    color: var(--orange);
    background: #11141c;
    border-radius: 13px;
    width: 56px;
    height: 56px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 27px;
}

.chat-title {
    font-size: 23px;
    font-weight: 800;
}

.chat-subtitle {
    color: #bdcce0;
    font-size: 14px;
    margin-top: 4px;
}

.stButton > button {
    background: linear-gradient(145deg, #101723, #0c121c);
    color: #eaf0f8;
    border: 1px solid #344257;
    border-radius: 12px;
    font-weight: 650;
    min-height: 46px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border-color: var(--orange);
    background: #251614;
    box-shadow: 0 0 14px #ff43282b;
}

.overview-card {
    background: linear-gradient(145deg, #131c29, #0e1520);
    border: 1px solid #202b3b;
    border-radius: 16px;
    padding: 22px;
    margin-top: 20px;
    margin-bottom: 20px;
}

.overview-heading {
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 12px;
}

.overview-body {
    color: #dce5f1;
    font-size: 14px;
    line-height: 1.8;
    white-space: pre-wrap;
}

div[data-testid="stChatInput"] {
    border: 1px solid #344258;
    border-radius: 14px;
    background: #172130;
}

div[data-testid="stChatInput"] textarea {
    color: white;
}

div[data-testid="stChatMessage"] {
    background: #111a27;
    border: 1px solid #202c3e;
    border-radius: 14px;
    padding: 12px;
}

.footer {
    border-top: 1px solid #283141;
    margin-top: 25px;
    padding: 18px 0 8px;
    text-align: center;
    color: #d4deea;
    font-size: 13px;
    word-spacing: 8px;
}

.stAlert {
    border-radius: 12px;
}

@media (max-width: 800px) {
    .block-container {
        padding: 1rem;
    }
    .hero-title {
        margin-top: 10px;
    }
    .chat-panel {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD RAG COMPONENTS ----------------
@st.cache_resource
def get_rag_components():
    # rag_engine.load_rag() returns (retriever, llm)
    return load_rag()


# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_car" not in st.session_state:
    st.session_state.selected_car = "Safari"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------------- PAGE LAYOUT ----------------
left, right = st.columns([0.78, 1.22], gap="large")

with left:
    st.markdown('<div class="hero-title">TATA CAR GUIDE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">RAG-Powered Automotive Chatbot</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="orange-line"></div>', unsafe_allow_html=True)

    st.markdown('''
    <div class="feature">
        <div class="feature-icon">▤</div>
        <div>
            <div class="feature-title">Tata Brochures Knowledge</div>
            <div class="feature-description">Retrieves information from your local Tata car PDF using Retrieval-Augmented Generation (RAG).</div>
        </div>
    </div>
    <div class="feature">
        <div class="feature-icon">✺</div>
        <div>
            <div class="feature-title">Smart &amp; Contextual Answers</div>
            <div class="feature-description">Uses LangChain and Groq to generate relevant, context-based responses.</div>
        </div>
    </div>
    <div class="feature">
        <div class="feature-icon">ϟ</div>
        <div>
            <div class="feature-title">Explore All Tata Models</div>
            <div class="feature-description">Ask about variants, features, colours, engines, electric powertrains, and more.</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Optional image: place your own image at assets/tata-safari.png
    if os.path.exists("assets/tata-safari.png"):
        st.image("assets/tata-safari.png", use_container_width=True)

with right:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    header_left, header_right = st.columns([4, 1.2])

    with header_left:
        st.markdown('''
        <div class="chat-header">
            <div class="chat-icon">▣</div>
            <div>
                <div class="chat-title">Tata Car Guide</div>
                <div class="chat-subtitle">Ask about variants, features, engines, batteries and colours.</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with header_right:
        if st.button("⊕ New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    # SUV buttons in two rows
    for row_index, row_cars in enumerate((CARS[:5], CARS[5:])):
        cols = st.columns(5, gap="small")
        for col, car in zip(cols, row_cars):
            with col:
                if st.button(car, key=f"car_{car}", use_container_width=True):
                    st.session_state.selected_car = car
                    st.session_state.pending_question = (
                        f"Give me a quick overview of the Tata {car} based on the PDF. "
                        "Include variants, prices, colours, engine or battery details only if provided."
                    )
                    st.rerun()

    # Load the retriever and model; errors are displayed in the UI
    try:
        retriever, llm = get_rag_components()
        load_error = None
    except Exception as error:
        retriever, llm = None, None
        load_error = str(error)

    # Initial overview card
    if not st.session_state.messages:
        st.markdown(
            f'<div class="overview-card"><div class="overview-heading">🚘 Tata {st.session_state.selected_car} — Quick Overview</div>',
            unsafe_allow_html=True,
        )
        if load_error:
            st.warning("Chatbot setup needs attention. Check your .env, ChromaDB, packages, and Groq API key.")
            with st.expander("Show setup error"):
                st.code(load_error)
        else:
            with st.spinner("Preparing the overview from your PDF..."):
                try:
                    overview_question = (
                        f"Give a concise quick overview of Tata {st.session_state.selected_car}. "
                        "Include only details in the PDF: price, variants, colours, engine, powertrain, and battery if applicable."
                    )
                    overview, overview_sources = ask_tata_question(
                        overview_question, retriever, llm
                    )
                    st.markdown(overview)
                    with st.expander("Overview PDF sources"):
                        for source in overview_sources:
                            st.write(f"Page {source['page']}: {source['text']}")
                except Exception as error:
                    st.error(f"Could not generate overview: {error}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("PDF sources"):
                    for source in message["sources"]:
                        st.write(f"Page {source['page']}: {source['text']}")

    # Chat input and SUV-button questions share the same RAG function
    user_question = st.chat_input("Ask about a Tata car, variant or feature...")
    if not user_question and st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        if load_error:
            st.error(f"Cannot start the chatbot: {load_error}")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching the PDF and asking Groq..."):
                    try:
                        # Correct function from rag_engine.py
                        response, sources = ask_tata_question(
                            user_question, retriever, llm
                        )
                        st.markdown(response)
                        with st.expander("PDF sources"):
                            for source in sources:
                                st.write(f"Page {source['page']}: {source['text']}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                        })
                    except Exception as error:
                        st.error(f"Error generating answer: {error}")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('''
<div class="footer">
    🐍 Python &nbsp; • &nbsp; 🔗 LangChain &nbsp; • &nbsp;
    🟣 ChromaDB &nbsp; • &nbsp; 🤗 Hugging Face &nbsp; • &nbsp; 🔥 Groq
</div>
''', unsafe_allow_html=True)
