from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import aiohttp
import fitz  # PyMuPDF
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import hashlib
import uuid
import asyncio

app = FastAPI()

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "rag_docs"

class ChatMessage(BaseModel):
    message: str
    source: str = "web"

async def embed_texts(texts):
    response = await openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [e.embedding for e in response.data]

def chunk_text(text, max_length=500):
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    chunks = chunk_text(full_text)
    vectors = await embed_texts(chunks)
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk}
        ))
    await qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "success", "chunks": len(chunks)}

async def search_qdrant(query: str):
    embedded = await embed_texts([query])
    result = await qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedded[0],
        limit=3
    )
    return [hit.payload["text"] for hit in result]

async def chat_with_openrouter(prompt):
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://yourdomain.com",
            "X-Title": "MyUpgrd Chatbot"
        }
        json_payload = {
            "model": "openrouter/mixtral-8x7b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_payload) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

@app.post("/chat")
async def chat(msg: ChatMessage):
    related_chunks = await search_qdrant(msg.message)
    context = "\n".join(related_chunks)
    full_prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {msg.message}"
    reply = await chat_with_openrouter(full_prompt)
    return JSONResponse(content={"reply": reply})

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    message = data.get("message", {}).get("text")
    chat_id = data.get("message", {}).get("chat", {}).get("id")
    if not message or not chat_id:
        return {"status": "ignored"}
    related_chunks = await search_qdrant(message)
    context = "\n".join(related_chunks)
    full_prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {message}"
    reply = await chat_with_openrouter(full_prompt)
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as session:
        await session.post(telegram_url, json={"chat_id": chat_id, "text": reply})
    return {"status": "replied"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
