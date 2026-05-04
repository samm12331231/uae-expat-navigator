import os
import pymupdf4llm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

SOURCE_DIR = "./data/pdfs"
CHROMA_PATH = "./data/chroma_db"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

AUTHORITY_MAP = {
    "RTA": "Roads and Transport Authority",
    "ICP": "Federal Authority for Identity, Citizenship, Customs & Port Security",
    "MOHRE": "Ministry of Human Resources and Emiratisation",
    "GDRFA": "General Directorate of Residency and Foreigners Affairs"
}

def ingest_docs():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    all_chunks = []

    for file in os.listdir(SOURCE_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(SOURCE_DIR, file)

            md_text = pymupdf4llm.to_markdown(path)
            chunks = splitter.split_text(md_text)

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

    print(f"\n✅ Done — {len(all_chunks)} total chunks in {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_docs()