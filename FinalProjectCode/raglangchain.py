import faiss
import json
from sentence_transformers import SentenceTransformer

# ---------------- Load FAISS index ----------------
index = faiss.read_index("legal_hnsw.index")

# ---------------- Load metadata ----------------
with open("legal_corpus_meta.jsonl", "r", encoding="utf-8") as f:
    meta = [json.loads(line.strip()) for line in f]

# ---------------- Custom retriever ----------------
class SimpleRetriever:
    def __init__(self, index, meta):
        self.index = index
        self.meta = meta

    def get_relevant_documents(self, query_vector, top_k=5):
        D, I = self.index.search(query_vector, top_k)
        results = []
        for i in I[0]:
            if i != -1:
                results.append(self.meta[i])
        return results

# ---------------- Embed query using your bi-encoder ----------------
bi_encoder = SentenceTransformer("law-ai/InLegalBERT")

def embed_query(text):
    vec = bi_encoder.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    return vec

retriever = SimpleRetriever(index, meta)

# ---------------- Retrieval Function ----------------
def retrieve_documents(query, top_k=5):
    q_vec = embed_query(query)
    docs = retriever.get_relevant_documents(q_vec, top_k=top_k)
    return docs

# ---------------- Example ----------------
query = "Can police enter my house without a warrant?"
results = retrieve_documents(query)

for i, doc in enumerate(results, 1):
    print(f"=== Document {i} ===")
    print(f"Source: {doc['source_pdf']}, Page: {doc['page']}, Clause: {doc.get('clause_label','N/A')}")
    print(f"Text: {doc['text']}\n")
