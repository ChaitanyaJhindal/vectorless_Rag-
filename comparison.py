import time
import os
import json
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return "{}"

def judge(query, context_a, answer_a, context_b, answer_b):
    prompt = f"""
You are an expert evaluator.

Question: {query}

Answer A:
{answer_a}

Context A:
{context_a}

Answer B:
{answer_b}

Context B:
{context_b}

Evaluate BOTH answers on:
1. Correctness (0-10)
2. Relevance (0-10)
3. Faithfulness (0-10) → Is answer grounded in context?
4. Hallucination (0-10) → Higher = MORE hallucination

Return ONLY JSON:

{{
  "A": {{
    "correctness": number,
    "relevance": number,
    "faithfulness": number,
    "hallucination": number
  }},
  "B": {{
    "correctness": number,
    "relevance": number,
    "faithfulness": number,
    "hallucination": number
  }},
  "winner": "A" or "B",
  "reason": "short reason"
}}
"""

    raw = llm.invoke(prompt)

    try:
        clean = extract_json(raw)
        return json.loads(clean)
    except:
        return {
            "error": True,
            "raw_output": raw[:300]
        }


loader = PyPDFLoader("Your Actual FIle Path ")
documents = loader.load()


pages = [{"page": i+1, "content": d.page_content} for i, d in enumerate(documents)]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")

if not os.path.exists("./chroma_db"):
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    db.persist()
else:
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever()

llm = OllamaLLM(model="gemma3:1b", temperature=0)


test_queries = [
    "What does penalty clause say?",
    "Which pages are referenced for liability?",
    "Explain default consequences with references",
    "What happens in case of violation?",
    "Find sections related to repayment and penalties"
]


for query in test_queries:
    print("\n==============================")
    print("Query:", query)

    # -------- PageIndex --------
    start = time.time()

    summaries = "\n".join([
        f"Page {p['page']}: {p['content'][:300]}"
        for p in pages
    ])

    selection_prompt = f"""
Query: {query}

{summaries}

Return JSON list of relevant page numbers.
"""

    try:
        selected = json.loads(extract_json(llm.invoke(selection_prompt)))
    except:
        selected = [1]

    context_page = "\n".join([
        p["content"] for p in pages if p["page"] in selected
    ])

    answer_page = llm.invoke(f"Context:\n{context_page}\nQuestion:{query}")
    time_page = round(time.time() - start, 2)

    # -------- Vector RAG --------
    start = time.time()

    docs_ret = retriever.invoke(query)
    context_vec = "\n".join([d.page_content for d in docs_ret])

    answer_vec = llm.invoke(f"Context:\n{context_vec}\nQuestion:{query}")
    time_vec = round(time.time() - start, 2)

    # -------- Judge --------
    scores = judge(query, context_page, answer_page, context_vec, answer_vec)

    # -------- Output --------
    print("\n--- PageIndex ---")
    print("Time:", time_page, "s")
    print(answer_page[:300])

    print("\n--- Vector RAG ---")
    print("Time:", time_vec, "s")
    print(answer_vec[:300])

    print("\n--- Evaluation ---")
    print(scores)