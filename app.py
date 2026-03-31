from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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