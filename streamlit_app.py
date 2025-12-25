import streamlit as st
from dotenv import load_dotenv

from ingest import get_youtube_transcript, build_faiss_index
from chatbot import start_chat  # optional if you want CLI reuse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥")

st.title("🎥 YouTube Video Chatbot")
st.caption("Ask questions based only on the video content")

# ---------- Session State ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Helper ----------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------- Sidebar ----------
st.sidebar.header("📌 Video Input")

video_id = st.sidebar.text_input(
    "Enter YouTube Video ID",
    placeholder="e.g. XF6DCrNTzug"
)

if st.sidebar.button("Load Video"):
    if not video_id.strip():
        st.error("Please enter a valid video ID")
    else:
        with st.spinner("Fetching transcript & building index..."):
            try:
                text = get_youtube_transcript(video_id)
                st.session_state.vectorstore = build_faiss_index(text)
                st.session_state.chat_history = []
                st.success("Video loaded successfully!")
            except Exception as e:
                st.error(str(e))

# ---------- Chat UI ----------
if st.session_state.vectorstore:

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    )

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    rag_chain = parallel_chain | prompt | llm | StrOutputParser()

    # Display history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # User input
    user_query = st.chat_input("Ask a question about the video...")

    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(user_query)

        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.info("👈 Enter a YouTube video ID from the sidebar to begin.")
