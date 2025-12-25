from dotenv import load_dotenv
load_dotenv()

from ingest import get_youtube_transcript, build_faiss_index
from chatbot import start_chat


def main():
    video_id = input("Enter YouTube video ID: ").strip()

    print("\n⏳ Fetching transcript...")
    text = get_youtube_transcript(video_id)

    print("⏳ Building FAISS index (local embeddings)...")
    vectorstore = build_faiss_index(text)

    print("✅ Chatbot ready!")
    start_chat(vectorstore)


if __name__ == "__main__":
    main()