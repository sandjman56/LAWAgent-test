from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai, os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentInput(BaseModel):
    text: str

@app.post("/spot_issues")
async def spot_issues(input: DocumentInput):
    try:
        prompt = f"""
You are acting as a litigation associate at a commercial litigation firm.

Identify all legal issues or causes of action raised in the document below.

- Focus on legal issues (e.g., breach of contract, misrepresentation).
- Give a 1–2 sentence explanation under each.
- Present as bullet points.

Document:
{input.text}
"""
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a skilled litigation attorney."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700
        )
        return {"issues": response['choices'][0]['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
