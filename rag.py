import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Load and prepare once
loader = PyPDFLoader("Your path file ")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")

# Create or load DB
if not os.path.exists("./chroma_db"):
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    db.persist()
else:
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever()
llm = OllamaLLM(model="gemma3:1b")

# Chat loop
while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"Answer based only on the context:\n{context}\n\nQuestion: {query}"

    response = llm.invoke(prompt)

    print("Bot:", response)