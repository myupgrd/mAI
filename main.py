import os
import aiohttp
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, CollectionStatus
from supabase import create_client, Client
from uuid import uuid4
from PyPDF2 import PdfReader
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL_MAI")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_MAI")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_USERNAME = "myupgrd_bot"  # Replace with your actual bot username

# Initialize clients
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
COLLECTION_NAME = "rag_docs"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure Qdrant collection exists
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

# Get OpenAI embedding
async def get_embedding(text):
    response = await client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Get previous chat history for memory
def get_chat_history(user_id, limit=5):
    try:
        response = supabase.table("chat_logs") \
            .select("message,response") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()

        history = response.data or []
        messages = []
        for entry in reversed(history):
            messages.append({"role": "user", "content": entry["message"]})
            messages.append({"role": "assistant", "content": entry["response"]})
        return messages
    except Exception as e:
        print("Supabase retrieval error:", e)
        return []

# Chat endpoint with memory and logging
async def chat_with_openrouter(prompt, user_id):
    embedded = await get_embedding(prompt)
    hits = qdrant.search(collection_name=COLLECTION_NAME, query_vector=embedded, limit=3)
    context = "\n\n".join([hit.payload.get("text", "") for hit in hits])

    memory_messages = get_chat_history(user_id)

    messages = [
        {
            "role": "system",
            "content": (
                "You are mAI — the intelligent, friendly assistant of myupgrd (myupgrd Artificial Intelligence). "
                "You respond in a fresh, professional, human-like tone. You're multilingual and help users navigate, learn, and act. "
                "If you know a link, provide it directly. Always include useful information, empathy, and clarity.\n\n"
                f"Context: {context}"
            )
        },
        *memory_messages,
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
            reply = data["choices"][0]["message"]["content"]

            try:
                supabase.table("chat_logs").insert({
                    "id": str(uuid4()),
                    "user_id": user_id,
                    "message": prompt,
                    "response": reply,
                    "timestamp": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                print("Supabase insert error:", e)

            return reply

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("message", "")
    user_id = body.get("user_id", "guest")
    if not prompt:
        return {"reply": "No message provided"}
    reply = await chat_with_openrouter(prompt, user_id)
    return {"reply": reply}

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

@app.get("/health")
def health():
    return {"status": "online"}

@app.post("/telegram")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" in data and "text" in data["message"]:
        message = data["message"]
        user_text = message["text"]
        chat_id = message["chat"]["id"]
        user_id = str(message["from"]["id"])

        if message.get("chat", {}).get("type") == "private" or f"@{BOT_USERNAME}" in user_text:
            cleaned_text = user_text.replace(f"@{BOT_USERNAME}", "").strip()
            reply = await chat_with_openrouter(cleaned_text, user_id)
            await send_telegram_message(chat_id, reply)

    return {"ok": True}

async def send_telegram_message(chat_id, text):
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as session:
        await session.post(telegram_url, json={"chat_id": chat_id, "text": text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
