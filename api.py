import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Search and Recommend API")

# lets the team's website call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")
book_vectors = np.load("book_vectors.npy")
with open("books.json", encoding="utf-8") as f:
    books = json.load(f)

MIN_SCORE = 0.30


def to_result(book, score):
    return {
        "id": book["id"],
        "title": book["title"],
        "author": book["author"],
        "score": round(float(score), 3),
    }


@app.get("/search")
def search(q: str, top_k: int = 5):
    query_vector = model.encode(q)
    scores = util.cos_sim(query_vector, book_vectors)[0]
    ranked = sorted(zip(books, scores), key=lambda x: -x[1])[:top_k]
    results = [to_result(b, s) for b, s in ranked if s >= MIN_SCORE]
    return {"query": q, "results": results}


@app.get("/recommend/{book_id}")
def recommend(book_id: int, top_k: int = 5):
    matches = [i for i, b in enumerate(books) if b["id"] == book_id]
    if not matches:
        raise HTTPException(status_code=404, detail="Book not found")
    idx = matches[0]

    scores = util.cos_sim(book_vectors[idx], book_vectors)[0]
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    results = []
    for i, score in ranked:
        if books[i]["title"] == books[idx]["title"]:
            continue  # skip the book itself and duplicate editions
        results.append(to_result(books[i], score))
        if len(results) == top_k:
            break
    return {"based_on": books[idx]["title"], "results": results}