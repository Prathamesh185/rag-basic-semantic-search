import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# Load environment variables
load_dotenv()

# Gemini Client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Embedding Model
encoder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)