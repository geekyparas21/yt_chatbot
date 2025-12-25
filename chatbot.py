from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def start_chat(vectorstore):
    # 1. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 2. Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely and try to make sure that the question is answer in a way that anybody can understand whether one knows something about the context or not.
"""
    )

    # 3. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Parallel chain (retrieval + question passthrough)
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    # 5. Output parser
    parser = StrOutputParser()

    # 6. Final chain
    rag_chain = parallel_chain | prompt | llm | parser

    print("\n🤖 YouTube Video Chatbot (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(question)
        print("\nBot:", answer, "\n")
