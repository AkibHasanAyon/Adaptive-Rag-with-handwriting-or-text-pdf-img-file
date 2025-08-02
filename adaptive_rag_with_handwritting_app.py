import os
import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import fitz  # PyMuPDF
import easyocr
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------- Groq API Init ------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------- Load Models --------------------------
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=False)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return reader, embedder

reader, embedding_model = load_models()

# ---------------------- OCR Functions ------------------------
def easyocr_image(image: Image.Image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    result = reader.readtext(np.array(image), detail=0, paragraph=True)
    return "\n".join(result)

@st.cache_data(show_spinner=False)
def extract_text(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        full_text = ""
        for i, page in enumerate(doc):
            typed_text = page.get_text().strip()
            if len(typed_text) > 50:
                full_text += typed_text + "\n"
            else:
                pix = page.get_pixmap(dpi=400)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                full_text += easyocr_image(img) + "\n"
        return full_text.strip()
    elif ext in [".jpg", ".jpeg", ".png"]:
        img = Image.open(file_path)
        return easyocr_image(img)
    else:
        return ""

# ---------------------- RAG Functions ------------------------
def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_indexes(chunks):
    embeddings = embedding_model.encode(chunks)
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return embeddings, bm25

def adaptive_retrieve(query, chunks, embeddings, bm25, top_k=3, alpha=0.5):
    query_embedding = embedding_model.encode([query])[0]
    vector_scores = cosine_similarity([query_embedding], embeddings)[0]
    keyword_scores = bm25.get_scores(query.split())

    # Normalize
    vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + 1e-8)
    keyword_scores = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores) + 1e-8)

    combined_scores = alpha * vector_scores + (1 - alpha) * keyword_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def ask_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ---------------------- Streamlit UI ------------------------
st.set_page_config(page_title="Medical RAG", page_icon="🩺", layout="centered")
st.title("🩺 Adaptive RAG on Medical Handwritten Files")
st.markdown("Upload a scanned **medical report** (PDF/Image) and ask your question below.")

uploaded_file = st.file_uploader("📤 Upload a handwritten medical file (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success("✅ File uploaded successfully.")
    with st.spinner("🔍 Extracting and chunking text..."):
        try:
            raw_text = extract_text(file_path)
            if not raw_text or len(raw_text) < 20:
                st.error("❌ OCR failed. Please upload a clearer image or scan.")
                st.stop()
            chunks = chunk_text(raw_text)
            embeddings, bm25 = build_indexes(chunks)
        except Exception as e:
            st.error(f"OCR or parsing failed: {e}")
            st.stop()
    st.success("✅ Ready to answer your questions.")

    query = st.text_input("💬 What would you like to know from this document?")
    if query:
        with st.spinner("💬 Thinking..."):
            retrieved = adaptive_retrieve(query, chunks, embeddings, bm25)
            context = "\n".join(retrieved)
            final_prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
            answer = ask_groq(final_prompt)
        st.markdown("### 🧠 Answer")
        st.write(answer)
