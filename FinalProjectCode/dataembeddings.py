"""
dataembeddings.py
Legal RAG pipeline: PDF extraction -> clause-aware chunking -> embeddings -> FAISS -> retrieval -> answer generation (without external LLM)
"""

import os
import re
import uuid
import json
import faiss
import pdfplumber
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.tokenize import sent_tokenize

# ---------------- CONFIG ----------------
PDF_FILES = [
    "dataaa/CrPC.pdf",
    "dataaa/THE POLICE ORDER, 2002.pdf",
    "dataaa/The Constitution of pak.pdf"
]
BI_ENCODER = "law-ai/InLegalBERT"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_TOP_K = 50
RERANK_TOP_K = 7
MAX_CHARS = 900
OVERLAP_CHARS = 250
MIN_CHARS = 40
FAISS_INDEX_PATH = "legal_hnsw.index"
META_JSONL_PATH = "legal_corpus_meta.jsonl"

# Clause pattern for legal texts
CLAUSE_PATTERN = re.compile(
    r"(Section\s+\d+[A-Za-z0-9\(\)\-\/]*|Article\s+\d+[A-Za-z0-9\(\)\-\/]*|S\.\s*\d+|Clause\s*\(?[a-zA-Z0-9]+\)?|Subsection\s*\(?\d+\)?|sub\-section\s*\(?\d+\)?)",
    flags=re.IGNORECASE
)

def find_clause_label(text):
    m = CLAUSE_PATTERN.search(text)
    return m.group(0) if m else None

# ---------------- PDF extraction ----------------
def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, p in enumerate(pdf.pages):
            txt = p.extract_text() or ""
            pages.append({"page_no": i+1, "text": txt})
    return pages

# ---------------- Clause-aware chunking ----------------
def chunk_text_clause_aware(text, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS, min_chars=MIN_CHARS):
    if not text.strip():
        return []
    sents = sent_tokenize(text)
    chunks, cur, char_pos = [], "", 0
    cur_start = 0
    for sent in sents:
        sent = sent.strip()
        if not cur:
            cur_start = char_pos
        if len(cur) + len(sent) + 1 > max_chars:
            if len(cur) >= min_chars:
                chunks.append((cur.strip(), cur_start, char_pos))
                overlap_text = cur[-overlap:] if overlap < len(cur) else cur
                cur = overlap_text + " " + sent
                cur_start = max(cur_start + len(cur) - len(overlap_text) - len(sent) - 1, char_pos - len(cur))
            else:
                cur = (cur + " " + sent).strip()
        else:
            cur = (cur + " " + sent).strip()
        char_pos += len(sent) + 1
    if cur and len(cur.strip()) >= min_chars:
        chunks.append((cur.strip(), cur_start, char_pos))
    # refine by clause labels
    refined = []
    for text_chunk, st, ed in chunks:
        labels = list(CLAUSE_PATTERN.finditer(text_chunk))
        if labels:
            first = labels[0]
            if 20 < first.start() < len(text_chunk) - 20:
                left = text_chunk[:first.start()].strip()
                right = text_chunk[first.start():].strip()
                if len(left) >= min_chars:
                    refined.append((left, st, st + first.start()))
                if len(right) >= min_chars:
                    refined.append((right, st + first.start(), ed))
                continue
        refined.append((text_chunk, st, ed))
    return refined

# ---------------- Build corpus and FAISS index ----------------
def build_corpus_and_index(pdf_files, bi_encoder_name=BI_ENCODER,
                           faiss_index_path=FAISS_INDEX_PATH,
                           meta_jsonl_path=META_JSONL_PATH):
    bi_encoder = SentenceTransformer(bi_encoder_name)
    corpus_texts, corpus_meta, embeddings = [], [], []
    for pdf in pdf_files:
        if not os.path.exists(pdf):
            print(f"Warning: {pdf} not found. Skipping.")
            continue
        pages = extract_pages(pdf)
        for p in tqdm(pages, desc=f"Chunking pages of {os.path.basename(pdf)}"):
            chunks = chunk_text_clause_aware(p["text"])
            for ch_text, ch_start, ch_end in chunks:
                meta = {
                    "id": str(uuid.uuid4()),
                    "source_pdf": os.path.basename(pdf),
                    "page": p["page_no"],
                    "char_start": int(ch_start),
                    "char_end": int(ch_end),
                    "clause_label": find_clause_label(ch_text)
                }
                corpus_texts.append(ch_text)
                corpus_meta.append(meta)
    if not corpus_texts:
        raise RuntimeError("No chunks generated from PDFs.")
    # embed chunks
    batch_size = 32
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding batches"):
        batch_texts = corpus_texts[i:i+batch_size]
        embs = bi_encoder.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False)
        embeddings.append(np.array(embs, dtype=np.float32))
    embeddings = np.vstack(embeddings)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    # Build HNSW index
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    faiss.write_index(index, faiss_index_path)
    # save metadata
    with open(meta_jsonl_path, "w", encoding="utf-8") as outf:
        for meta, text in zip(corpus_meta, corpus_texts):
            out = meta.copy()
            out["text"] = text
            outf.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Corpus built. Chunks: {len(corpus_texts)}")
    return {"faiss_index_path": faiss_index_path, "meta_jsonl_path": meta_jsonl_path}

# ---------------- Load index and metadata ----------------
def load_index_and_meta(faiss_index_path=FAISS_INDEX_PATH, meta_jsonl_path=META_JSONL_PATH):
    if not os.path.exists(faiss_index_path) or not os.path.exists(meta_jsonl_path):
        raise RuntimeError("Index or metadata not found. Build corpus first.")
    index = faiss.read_index(faiss_index_path)
    meta = [json.loads(line.strip()) for line in open(meta_jsonl_path, "r", encoding="utf-8")]
    return index, meta

# ---------------- Retrieval ----------------
def retrieve(query, index, meta, bi_encoder_name=BI_ENCODER, reranker_name=RERANKER,
             dense_top_k=DENSE_TOP_K, rerank_top_k=RERANK_TOP_K):
    bi_encoder = SentenceTransformer(bi_encoder_name)
    cross = CrossEncoder(reranker_name, device="cpu")
    q_emb = bi_encoder.encode(query, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = index.search(q_emb.reshape(1, -1), dense_top_k)
    candidate_idxs = [int(i) for i in I[0] if i != -1]
    if not candidate_idxs:
        return []
    pairs = [(query, meta[idx]["text"]) for idx in candidate_idxs]
    scores = cross.predict(pairs)
    candidates = [{"score": float(s), **meta[idx]} for idx, s in zip(candidate_idxs, scores)]
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected, seen_texts = [], set()
    for c in candidates:
        txt = c["text"].strip()
        if txt in seen_texts:
            continue
        seen_texts.add(txt)
        selected.append(c)
        if len(selected) >= rerank_top_k:
            break
    return selected

# ---------------- Generate answer from passages (no external LLM) ----------------
def generate_answer_from_passages(passages):
    if not passages:
        return "No relevant passages found."
    answer = []
    for i, p in enumerate(passages, start=1):
        citation = f"[{p.get('source_pdf','?')} p.{p.get('page','?')} clause:{p.get('clause_label','N/A')}]"
        answer.append(f"{i}. {p['text'].strip()} {citation}")
    return "\n\n".join(answer)

# ---------------- Main ----------------
if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(META_JSONL_PATH):
        print("Building corpus and FAISS index...")
        build_corpus_and_index(PDF_FILES)
    else:
        print("FAISS index and metadata already exist. Skipping build.")

    index, meta = load_index_and_meta()

    query = "Can police enter my house without a warrant?"
    print("Retrieving for query:", query)
    results = retrieve(query, index, meta)

    for i, r in enumerate(results, start=1):
        print(f"\n--- Result {i} (score {r['score']:.4f}) ---")
        print(f"Source: {r['source_pdf']} Page: {r['page']} Clause: {r.get('clause_label')}")
        print(r['text'][:1000])

    print("\n=== Generated Answer from retrieved passages ===\n")
    answer = generate_answer_from_passages(results)
    print(answer)
