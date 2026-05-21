import time
import ollama

# ---------- CONFIG ----------
MODEL_NAME = "qwen3:4b"   # a tiny model that runs on anything; change to "llama3.2:3b" or "qwen2:0.5b" as you prefer
# ----------------------------

def test_llm():
    query = "capital of india"
    print(f"🔄 Sending test query to '{MODEL_NAME}' ...")
    start = time.perf_counter()

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.1}
        )
        answer = response["message"]["content"]
        elapsed = time.perf_counter() - start
        print(f"✅ Response received in {elapsed:.2f}s:\n{answer}")
    except Exception as e:
        print(f"❌ Ollama call failed: {e}")

if __name__ == "__main__":
    print("Ollama Local LLM Quick Test")
    test_llm()

    
# import ollama
# prompt = "How many years of imprisonment and how much fine should we pay if we take dowry?"
# resp = ollama.chat(
#     model="qwen3:1.7b",
#     messages=[
#         {"role": "system", "content": "You are a legal keyword extractor. Given a user question, return ONLY a comma-separated list of the most important legal keywords, acronyms, act names, or phrases. Do not include common words."},
#         {"role": "user", "content": prompt},
#     ],
#     options={"temperature": 0.0, "num_predict": 80},
# )
# print(f"Extracted legal keywords: {resp['message']['content']}")

# from keybert import KeyBERT
# kw_model = KeyBERT(model="all-MiniLM-L6-v2")
# query = "Best restaurants for vegan tacos in Hyderabad near Hi-tech city"
# keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), top_n=5)
# print(keywords)

# import ollama

# # The actual user query – keep it clean
# user_query = "How many years of imprisonment and how much fine should we pay if we take dowry?"

# # Prompt with clear instructions and a one-shot example
# prompt = f"""Extract the most important legal keywords, phrases, or act names from a user question.
# Return ONLY a comma-separated list of keywords. Do not include common words like 'what', 'how', 'the', etc.

# Example:
# Question: What is the punishment for murder under IPC?
# Keywords: murder, punishment, IPC, imprisonment, fine

# Now extract from the question below:
# Question: {user_query}
# Keywords:"""

# # Run the model with zero temperature for deterministic output
# response = ollama.generate(
#     model="sroecker/nuextract-tiny-v1.5",
#     prompt=prompt,
#     options={"temperature": 0.0}
# )

# # Print the raw output – you'll need to parse this list later
# print("Raw keywords output:", response['response'])



# from qdrant_client import QdrantClient
# client = QdrantClient(url="http://localhost:6333")
# hits = client.query_points(
#     collection_name="bilingual_hybrid_final",
#     query="demanding dowry fine ten thousand rupees",
#     limit=5,
#     with_payload=True,
# )
# for p in hits.points:
#     print(p.payload["filename"], p.payload["page"], p.payload["text"][:200])