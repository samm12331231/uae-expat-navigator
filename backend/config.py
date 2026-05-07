import os
from dotenv import load_dotenv

load_dotenv()

# Paths
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
SOURCE_DIR = os.getenv("SOURCE_DIR", "./data/pdfs")

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "llama-3.3-70b-versatile"

# Mapping for ingestion
AUTHORITY_MAP = {
    "RTA": "Roads and Transport Authority",
    "ICP": "Federal Authority for Identity, Citizenship, Customs & Port Security",
    "MOHRE": "Ministry of Human Resources and Emiratisation",
    "GDRFA": "General Directorate of Residency and Foreigners Affairs",
}

# Countries that get the South-Asian alternative path
SOUTH_ASIAN_COUNTRIES = [
    "India", "Pakistan", "Philippines", "Bangladesh", "Sri Lanka", "Nepal"
]