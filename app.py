from flask import Flask, request, jsonify, render_template

from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import os

print("Files in directory:", os.listdir())

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODELS ---------------- #

print("Loading models...")

index = faiss.read_index("faiss_index.index")

with open("chunks.json", "r") as f:
    chunks = json.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model2.to(device)

print("Models loaded successfully!")

# ---------------- GENERATION FUNCTION ---------------- #

def generate_answer(query, context_docs):
    context_text = "\n\n".join(context_docs)


# Make responses more human like and empathetic
    prompt = f"""
You are a kind and supportive maternal health assistant.

Always:
- Be warm, calm, and reassuring
- Use simple language
- Give practical advice
- Avoid sounding robotic

If the question is serious, gently encourage consulting a healthcare professional.

Question: {query}

Context:
{context_text}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model2.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# ---------------- ROUTE ---------------- #
@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/chatpage")
def chat_page():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"response": "Please enter a question."})

    # Step 1: Embed query
    query_embedding = embed_model.encode([user_query])

    # Step 2: Search FAISS
    k = 5
    distances, indices = index.search(query_embedding, k)

    # Step 3: Take top 3 most relevant
    top_k = 3
    selected_indices = indices[0][:top_k]

    retrieved_texts = [chunks[idx]["text"][:300] for idx in selected_indices]

    # Step 4: Generate answer
    answer = generate_answer(user_query, retrieved_texts)

    # Step 5: Get sources
    sources = list(set(
        chunks[idx].get("metadata", {}).get("source", "Unknown")
        for idx in selected_indices
    ))
    
    # Safety layer
    if "emergency" in user_query.lower():
        return jsonify({
            "response": "⚠️ This may be serious. Please seek immediate medical attention or contact a healthcare provider.",
            "sources": []
        })

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)