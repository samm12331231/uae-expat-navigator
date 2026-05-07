from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embeddings
)

queries = [
    "How do I convert my foreign driving license in Dubai?",
    "What documents do I need for a residence permit?",
    "How do I change my visa status?"
]

for query in queries:
    print(f"\n🔍 {query}")
    results = vectorstore.similarity_search(query, k=2)
    for r in results:
        print(f"  → {r.metadata['source']}: {r.page_content[:150]}")