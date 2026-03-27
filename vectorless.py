import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM

# 1. Load PDF
loader = PyPDFLoader("Your file path ")
documents = loader.load()

# 2. Create PageIndex (store pages + metadata)
pages = []
for i, doc in enumerate(documents):
    pages.append({
        "page": i + 1,
        "content": doc.page_content
    })

# 3. Load LLM
llm = OllamaLLM(model="gemma3:1b")

# 4. Chat loop
while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        break

    # Step 1: Create short summaries (first 100 chars of each page)
    summaries = "\n".join([
        f"Page {p['page']}: {p['content'][:100]}"
        for p in pages
    ])

    # Step 2: Ask LLM to pick relevant pages
    selection_prompt = f"""
    Query: {query}

    Below are page summaries:
    {summaries}

    Return only the page numbers that are most relevant (comma separated).
    """

    selected_pages = llm.invoke(selection_prompt)

    # Step 3: Extract relevant pages
    context = ""
    for p in pages:
        if str(p["page"]) in selected_pages:
            context += p["content"] + "\n"

    # Step 4: Final answer
    final_prompt = f"""
    Answer based only on the context:

    {context}

    Question: {query}
    """

    answer = llm.invoke(final_prompt)

    print("Bot:", answer) 