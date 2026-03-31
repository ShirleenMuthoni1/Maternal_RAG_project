from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load chunks
with open("chunks.json", "r") as f:
    chunks = json.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2") 

# Load flan-T5 generator
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("message")

    # This is where your RAG pipeline goes
    # retrieve → rerank → generate

    query_embedding = embed_model.encode([user_query])

    k = 5
    distances, indices = index.search(query_embedding, k)

    retrieved_texts = [chunks[idx]["text"] for idx in indices[0]]

    answer = generate_answer(user_query, retrieved_texts)

    return jsonify({
    "response": answer,
    "sources": [chunks[idx]["metadata"]["source"] for idx in indices[0]]
})


if __name__ == "__main__":
    app.run(debug=True)