import os
import aiohttp
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from PyPDF2 import PdfReader
from uuid import uuid4
import tiktoken
from openai import AsyncOpenAI

# ✅ Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["https://myupgrd.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Environment vars
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COLLECTION_NAME = "rag_docs"

# ✅ Qdrant setup
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ✅ Health check
@app.get("/health")
async def health():
    return {"status": "online"}

# ✅ Embedding function (new SDK)
async def get_embedding(text):
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

# ✅ Upload PDF and split into embeddings
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(raw_text)
    chunks = [tokens[i:i + 500] for i in range(0, len(tokens), 500)]

    points = []
    for chunk in chunks:
        text_chunk = tokenizer.decode(chunk)
        embedding = await get_embedding(text_chunk)
        points.append(PointStruct(id=str(uuid4()), vector=embedding, payload={"text": text_chunk}))

    await qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    return {"status": "success", "chunks": len(chunks)}

# ✅ Chat endpoint with RAG and OpenRouter call
@app.post("/chat")
async def chat(req: Request):
    try:
        data = await req.json()
        question = data.get("message", "")

        # Get question embedding
        question_embedding = await get_embedding(question)

        # Query Qdrant for context
        search_result = await qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=3
        )
        context = "\n".join([hit.payload["text"] for hit in search_result])

        prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        reply = await chat_with_openrouter(prompt)

        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error occurred: {str(e)}"}

# ✅ OpenRouter GPT-3.5 response handler
async def chat_with_openrouter(prompt):
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://myupgrd.com",
            "X-Title": "MyUpgrd Chatbot"
        }

        json_payload = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_payload) as resp:
            data = await resp.json()

            if "choices" not in data:
                print("❌ OpenRouter Error:", data)
                return f"Error from OpenRouter: {data.get('error', 'unknown error')}"

            return data["choices"][0]["message"]["content"]
