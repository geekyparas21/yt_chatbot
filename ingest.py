from dotenv import load_dotenv
load_dotenv()
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def detect_language(text: str) -> str:
    """
    Detect language of transcript text
    """
    try:
        return detect(text)
    except:
        return "unknown"


def translate_hindi_to_english(text: str) -> str:
    """
    Translate Hindi transcript to English using Gemini
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = f"""
Translate the following Hindi text into clear English.
Preserve meaning. Do not summarize.

Text:
{text}
"""

    return llm.invoke(prompt).content


def get_youtube_transcript(video_id: str) -> str: 
    """ 
    Fetch transcript text from a YouTube video 
    """

    try: 
        ytt_api = YouTubeTranscriptApi() 
        transcript = ytt_api.fetch(video_id,languages=["en","hi"]) 

        # IMPORTANT: use attribute access 
        raw_text = " ".join(chunk.text for chunk in transcript) 
        language = detect_language(raw_text) 
        print(f"📌 Detected transcript language: {language}") 

        if language == "hi": 
            print("🔄 Translating Hindi transcript to English...") 
            return translate_hindi_to_english(raw_text) 
        
        return raw_text 
    
    except TranscriptsDisabled: 
        raise RuntimeError("❌ No captions available for this video")
    

def build_faiss_index(text: str):
    """
    Split transcript and build FAISS vector store
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
