import os
import aiohttp
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, CollectionStatus
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

app = FastAPI()
client = AsyncOpenAI(api_key=OPENROUTER_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "rag_docs"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create collection if not exists
try:
    if not qdrant.get_collection(COLLECTION_NAME).status == CollectionStatus.GREEN:
        qdrant.recreate_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
except Exception:
    qdrant.recreate_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# Get embedding for text
async def get_embedding(text):
    response = await client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# RAG Chat with OpenRouter
async def chat_with_openrouter(prompt):
    # Search context from Qdrant
    embedded = await get_embedding(prompt)
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedded,
        limit=3
    )

    context = "\n\n".join([hit.payload.get("text", "") for hit in hits])

    messages = [
        {
            "role": "system",
            "content": (
                "You are mAI — the intelligent, friendly assistant of myupgrd (short for myupgrd Artificial Intelligence). "
                "You always respond in a human-like, conversational tone that feels fresh, helpful, and professional. "
                "You adapt your language based on the user's input — you're fluent in all major languages and reply in the language the user uses. "
                "You're a reliable guide and always ready to help, no matter the topic. "
                "Whenever possible, provide direct links to the requested pages or resources if you know the path. "
                "Always aim to be clear, kind, and helpful — like a knowledgeable friend with expert-level understanding. "
                "You also improve with every conversation by learning what users are asking for, refining your answers to be better each time. "
                "Your mission is to make every user feel heard, supported, and empowered on their journey with myupgrd.\n\n"
                f"Context:\n{context}"
            )
        },
        {"role": "user", "content": prompt}
    ]

    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://myupgrd.com",
            "X-Title": "MyUpgrd Chatbot"
        }

        payload = {
            "model": "openai/gpt-3.5-turbo",
            "messages": messages
        }

        async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as res:
            data = await res.json()
            if "choices" not in data:
                return f"Error occurred: {data}"
            return data["choices"][0]["message"]["content"]

# Endpoint: Chat
@app.post("/chat")
async def chat_endpoint(body: dict):
    prompt = body.get("message", "")
    if not prompt:
        return {"reply": "No message provided"}
    reply = await chat_with_openrouter(prompt)
    return {"reply": reply}

# Endpoint: Upload file and ingest to Qdrant
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = ""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        for page in reader.pages:
            content += page.extract_text() + "\n"
    else:
        content = (await file.read()).decode()

    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

    points = []
    for chunk in chunks:
        embedding = await get_embedding(chunk)
        point = PointStruct(id=str(uuid4()), vector=embedding, payload={"text": chunk})
        points.append(point)

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "success", "chunks": len(chunks)}

# Endpoint: Health check
@app.get("/health")
def health():
    return {"status": "online"}

# Run with uvicorn when deployed locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
