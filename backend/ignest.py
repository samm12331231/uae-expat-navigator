import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm

from .config import AUTHORITY_MAP, CHROMA_PATH, EMBEDDING_MODEL, SOURCE_DIR

load_dotenv()

EMBEDDINGS = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def ingest_docs():
    # Clear old DB to avoid duplicates on re-run
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️  Cleared old ChromaDB")

    # Header-aware but with size limits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for file in sorted(os.listdir(SOURCE_DIR)):
        if file.endswith(".pdf"):
            path = os.path.join(SOURCE_DIR, file)
            md_text = pymupdf4llm.to_markdown(path)
            chunks = splitter.create_documents([md_text])

            authority = next(
                (v for k, v in AUTHORITY_MAP.items() if k in file.upper()),
                "Unknown"
            )

            for chunk in chunks:
                chunk.metadata.update({
                    "source": file,
                    "authority": authority,
                    "type": "official_government_doc"
                })

            all_chunks.extend(chunks)
            print(f"✓ {file} → {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=EMBEDDINGS,
        persist_directory=CHROMA_PATH
    )
    _ = vectorstore  # persist_directory handles saving automatically

    print(f"\n✅ Done — {len(all_chunks)} total chunks in {CHROMA_PATH}")


if __name__ == "__main__":
    ingest_docs()
