import os
from pathlib import Path

import pytesseract
import requests
from pdf2image import convert_from_path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def build_vector_database():
    documents = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        images = convert_from_path(str(pdf_path))
        for page_number, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_number},
                    )
                )

    for text_path in DATA_DIR.glob("*.txt"):
        documents.extend(TextLoader(str(text_path), encoding="utf-8").load())

    if not documents:
        return None

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", " ", ""],
    ).split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)


faiss_vector_db = build_vector_database()


def search_ncair_knowledge_base(query: str, top_k: int = 2) -> str:
    if not faiss_vector_db:
        return "No knowledge base documents currently indexed."

    documents = faiss_vector_db.similarity_search(query, k=top_k)
    passages = []
    for document in documents:
        source = document.metadata.get("source", "Knowledge Base")
        page = document.metadata.get("page")
        heading = f"[{source} - Page {page}]" if page else f"[{source}]"
        passages.append(f"{heading}\n{document.page_content.strip()}")
    return "\n---\n".join(passages)


PORTAL_URL_REGISTRY = {
    "main": "https://lms.ncair.nitda.gov.ng",
    "login": "https://lms.ncair.nitda.gov.ng/intern/signin",
    "signin": "https://lms.ncair.nitda.gov.ng/intern/signin",
    "ncair_home": "https://ncair.nitda.gov.ng/",
    "register": "https://lms.ncair.nitda.gov.ng",
    "profile": "https://lms.ncair.nitda.gov.ng/intern/profile",
    "courses": "https://lms.ncair.nitda.gov.ng/intern/courses",
    "track_selection": "https://lms.ncair.nitda.gov.ng/intern/courses",
    "support": "https://ncair.nitda.gov.ng/contact/",
}


def get_portal_link(action: str) -> str:
    key = action.lower().strip()
    if key in PORTAL_URL_REGISTRY:
        title = key.replace("_", " ").title()
        return f"[Click here to open the {title} Page]({PORTAL_URL_REGISTRY[key]})"
    return f"[Click here to visit the NCAIR LMS Landing Page]({PORTAL_URL_REGISTRY['main']})"


def get_step_guidance(step_number: int) -> str:
    guidance = {
        1: "**Step 1 (Profile Setup):** Navigate to 'Edit Profile', complete all required bio fields, and save changes.",
        2: "**Step 2 (ID Verification):** Go to 'Documents', upload a valid ID under 2MB, and submit.",
        3: "**Step 3 (Track Selection):** Navigate to 'Track Selection' and choose your assigned department.",
        4: "**Step 4 (Final Submission):** Verify all details on your summary dashboard and click 'Submit Final Registration'.",
    }
    return guidance.get(step_number, "Invalid step number. Please specify a step between 1 and 4.")


def query_local_llama(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def agentic_rag_orchestrator(user_query: str) -> dict:
    query = user_query.lower()

    if any(word in query for word in ["link", "url", "where to", "open", "page", "access", "portal"]):
        if "register" in query or "sign up" in query:
            answer = f"To begin registration:\n\n{get_portal_link('register')}"
        elif "log" in query or "sign in" in query:
            answer = f"To log in:\n\n{get_portal_link('login')}"
        elif "profile" in query:
            answer = get_portal_link("profile")
        elif "track" in query:
            answer = get_portal_link("track_selection")
        elif "support" in query or "help" in query:
            answer = get_portal_link("support")
        else:
            answer = get_portal_link("main")
        return {"answer": answer, "tool": "get_portal_link", "evidence": "NCAIR portal registry"}

    if "stuck" in query or "step" in query:
        if "1" in query or "profile" in query:
            step = 1
        elif "2" in query or "upload" in query or "id" in query:
            step = 2
        elif "3" in query or "track" in query:
            step = 3
        elif "4" in query or "submit" in query:
            step = 4
        else:
            answer = "Complete onboarding sequence:\n\n" + "\n\n".join(get_step_guidance(i) for i in range(1, 5))
            return {"answer": answer, "tool": "get_step_guidance", "evidence": "NCAIR onboarding steps"}
        return {"answer": get_step_guidance(step), "tool": "get_step_guidance", "evidence": "NCAIR onboarding steps"}

    context = search_ncair_knowledge_base(user_query, top_k=2)
    prompt = f"""
You are the official NCAIR LMS Onboarding Assistant.
Answer the intern's question concisely using ONLY the official context below.
If it is insufficient, tell them to contact support at support@ncair.nitda.gov.ng.

--- OFFICIAL NCAIR CONTEXT ---
{context}
--------------------------------

User Question: {user_query}
Answer:
"""
    return {
        "answer": query_local_llama(prompt),
        "tool": "search_ncair_knowledge_base",
        "evidence": context,
    }
