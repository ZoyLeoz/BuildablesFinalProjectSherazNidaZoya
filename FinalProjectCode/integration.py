# app.py
import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

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

import torch
torch.set_default_dtype(torch.float32)

bi_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")



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

# ---------------- Configure Google API ----------------
with open("google_api_key.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()
genai.configure(api_key=GOOGLE_API_KEY)

model_name = "models/gemini-2.5-flash"
model = genai.GenerativeModel(model_name)


# ---------------- Streamlit UI ----------------
st.title("Pakistan Law and Advisory Assistance System")

user_query = st.text_input("Ask a legal question:")

if st.button("Get Answer") and user_query.strip() != "":
    st.info("Retrieving most relevant documents...")
    
    # Get ONLY top 3 most relevant docs
    results = retrieve_documents(user_query, top_k=5)

    # Always show the MOST relevant first (your retriever already returns in order)
    # No reordering needed unless FAISS changes, but this keeps your structure intact.

    for i, doc in enumerate(results, 1):

        article_info = f"Source: {doc['source_pdf']}, Page: {doc['page']}, Clause: {doc.get('clause_label','N/A')}"

        # NEW PROMPT → smooth, relevant, human-readable explanation
        prompt = (
            f"You are a legal assistant for pakistani Law. Your Job is to take the documents given to You and Provide a clean response on the Basis of the text in the document. DONOT give generalized text. Be specific and clear about the Laws.\n"
            f"User Query: {user_query}\n"
            f"Relevant Law Extract:\n{doc['text']}\n\n"
            f"Instructions:\n"
            f"- If A retrieved law does not represnet or answers the Query asked, donot provide its Explanation. No need to explicitly mention that this docmuent was not relevant. \n"
            f"- Ignore irrelevant parts.\n"
            f"- Explain ONLY the part relevant to the user's query.\n"
            f"- Write a clear legal explanation in simple English.\n"
            f"- Do NOT show the raw document text.\n"
            f"- Do NOT Generalize the Document Law. .\n"
            f"- At the end, add: (Reference: {article_info}) iff you are providing explanation for that document\n"
        )

        st.info(f"-------------")
        response = model.generate_content(prompt)

        # Show ONLY the explanation, NOT the raw document
        st.markdown(response.text)
