import os
import json
import re
import random
import string
import numpy as np
import pdfplumber
import streamlit as st
import io
from gtts import gTTS
# LangChain / FAISS
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from rank_bm25 import BM25Okapi

# Gemini
import google.generativeai as genai

# -------------------------------
# Secrets and initialization
# -------------------------------
# In Community Cloud, set:
# .streamlit/secrets.toml:
# GEMINI_API_KEY = "..."
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -------------------------------
# Gemini LLM Wrapper
# -------------------------------
class GeminiLLM:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Error calling Gemini API: {e}]"

# -------------------------------
# PDF Text Extractor
# -------------------------------
def extract_text_from_pdf(file_like) -> str:
    text = ""
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# -------------------------------
# Conversational RAG Agent
# -------------------------------
class ConversationalRAGAgent:
    def __init__(self, retriever_fn, llm: GeminiLLM, alpha: float = 0.6, top_k: int = 3):
        self.retriever_fn = retriever_fn
        self.llm = llm
        self.alpha = alpha
        self.top_k = top_k

    def _build_prompt(self, question, retrieved_docs):
        context = "\n---\n".join(retrieved_docs)
        prompt = (
            "You are a helpful assistant. Use the following context from company policy documents to answer.\n"
            f"Context:\n{context}\n\n"
            "Now the user asks:\n"
            f"{question}\n"
            "Answer strictly based on the context above. If you do not know, say you don't know.\n"
        )
        return prompt

    def ask(self, query):
        retrieved = self.retriever_fn(query, k=self.top_k, alpha=self.alpha)
        retrieved_docs = [doc for doc, _ in retrieved]
        prompt = self._build_prompt(query, retrieved_docs)
        response = self.llm.generate(prompt)
        return response, retrieved_docs

# -------------------------------
# Resume Scorer Node
# -------------------------------
# -------------------------------
# Resume Scorer Node (Two-Step Stage 1)
# -------------------------------
def check_minimum_qualifications(resume_files, job_desc_file_like, gemini_api_key: str):
    genai.configure(api_key=gemini_api_key)
    llm = genai.GenerativeModel("gemini-2.0-flash")
    job_description = extract_text_from_pdf(job_desc_file_like)
    qualified_resumes = []

    for uploaded in resume_files:
        resume_text = extract_text_from_pdf(uploaded)

        # -------- Step 1.1: Extract structured info from resume --------
        extract_prompt = f"""
You are an expert HR assistant. Extract structured candidate information from this resume.

Resume:
{resume_text}

Instructions:
- Output ONLY a JSON with fields:
  - "name": candidate's full name
  - "email": candidate's email
  - "degree": full degree name
  - "passing_year": year of graduation
  - "cgpa": candidate CGPA (if available)
  - "experience": total years of relevant experience
"""
        resp_extract = llm.generate_content(extract_prompt)
        out_extract = resp_extract.candidates[0].content.parts[0].text

        try:
            m = re.search(r"\{.*\}", out_extract, flags=re.DOTALL)
            if m:
                candidate_info = json.loads(m.group(0))
            else:
                continue
        except Exception:
            continue

        # -------- Step 1.2: Check minimum requirements against JD --------
        check_prompt = f"""
You are an expert HR assistant. Decide if the candidate meets the minimum requirements for the job.

Job Description:
{job_description}

Candidate Info:
{json.dumps(candidate_info)}

Minimum Requirements Rules:
1) If the job description mentions a passing year for a certain level like University, the candidate's passing year must be less than or equal to that year.
2) If the job description mentions a minimum CGPA, the candidate's CGPA must be greater than or equal to that value.
3) If the job description mentions minimum years of experience, the candidate's total relevant experience must be greater than or equal to that value.

Instructions:
- Output ONLY a JSON with fields:
  - "name": candidate full name
  - "email": candidate email
  - "min_score": 10 if all applicable minimum requirements are met, 0 otherwise
"""
        resp_check = llm.generate_content(check_prompt)
        out_check = resp_check.candidates[0].content.parts[0].text

        try:
            m2 = re.search(r"\{.*\}", out_check, flags=re.DOTALL)
            if m2:
                data = json.loads(m2.group(0))
                if data.get("min_score", 0) > 0:
                    data["resume_text"] = resume_text
                    qualified_resumes.append(data)
        except Exception:
            continue

    return qualified_resumes


def score_qualified_resumes(qualified_resumes, job_desc_file_like, gemini_api_key: str):
    genai.configure(api_key=gemini_api_key)
    llm = genai.GenerativeModel("gemini-2.0-flash")
    job_description = extract_text_from_pdf(job_desc_file_like)
    scored = []
    for cand in qualified_resumes:
        resume_text = cand["resume_text"]
        prompt =f"""
You are an expert HR assistant. Score the following resume against this job description.

Job Description:
{job_description}

Resume:
{resume_text}

Scoring rules:
- Total score = 100
   - 10: minimum requirements (already passed)
   - 20: experience or achievements
   - 70: relevance to JD

Instructions:
- Output ONLY a JSON with fields: "name", "email", "score"
"""
        resp = llm.generate_content(prompt)
        out = resp.candidates[0].content.parts[0].text
        try:
            m = re.search(r"\{.*\}", out, flags=re.DOTALL)
            if m:
                d = json.loads(m.group(0))
                if not d.get("email"):
                    d["email"] = cand.get("email", "")
                scored.append(d)
        except Exception:
            continue
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
def generate_password(length=12):
    chars = string.ascii_letters + string.digits + "!@#$%^&*()"
    return ''.join(random.choice(chars) for _ in range(length))

def mock_create_account(first_name, last_name, department, personal_email):
    company_email = f"{first_name.lower()}.{last_name.lower()}@company.com"
    temp_password = generate_password()
    return {
        "company_email": company_email,
        "temporary_password": temp_password,
        "status": "Account created",
        "assigned_department": department,
        "personal_email": personal_email
    }
class OnboardingAgent:
    def __init__(self, llm: GeminiLLM, vectorstore, chunks,
                 first_name, last_name, department, personal_email,
                 retriever_fn=None, alpha: float = 0.6, top_k: int = 3):
        self.llm = llm
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.first_name = first_name
        self.last_name = last_name
        self.employee_name = f"{first_name} {last_name}"
        self.credentials = mock_create_account(first_name, last_name, department, personal_email)
        self.retriever_fn = retriever_fn
        self.alpha = alpha
        self.top_k = top_k

    def retrieve_all_policies(self) -> str:
        all_docs = self.vectorstore.similarity_search("company policies", k=len(self.chunks))
        policy_texts = [d.page_content for d in all_docs]
        return "\n---\n".join(policy_texts)

    def build_welcome_message(self, lang: str = "en"):
        welcome = f"Welcome aboard, {self.employee_name}! 🎉 We're excited to have you in {self.credentials['assigned_department']}."
        creds = (
            f"Here are your credentials:\n"
            f"- Company Email: {self.credentials['company_email']}\n"
            f"- Temporary Password: {self.credentials['temporary_password']}\n"
        )
        background = (
            "Our company has a proud history of innovation and growth. "
            "Our mission is to deliver outstanding solutions that empower people worldwide. "
            "We value collaboration, integrity, and continuous learning.\n"
        )
        policies = self.retrieve_all_policies()
        message = (
            f"{welcome}\n\n{creds}\n{background}"
            f"Here are the key company policies:\n{policies}\n\n"
            f"Feel free to ask me any questions!"
        )
    
        # Convert text to speech
        tts = gTTS(text=message, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
    
        return message, fp.read()

    def answer(self, user_input: str) -> str:
        if self.retriever_fn:
            retrieved_docs = [doc for doc, _ in self.retriever_fn(user_input, k=self.top_k, alpha=self.alpha)]
            context = "\n---\n".join(retrieved_docs)
        else:
            context = "Company policy context not available."
        prompt = (
            f"You are an onboarding assistant for a new employee.\n"
            f"Use the following company policy context to answer the employee's question.\n"
            f"Context:\n{context}\n\n"
            f"Employee asks: {user_input}\n"
            f"Answer clearly, politely, and strictly based on the context. "
            f"If the answer is not in context, politely say you don't know."
        )
        return self.llm.generate(prompt)

    def answer_with_audio(self, user_input: str, lang: str = "en"):
        """Answer and return (text, audio_bytes) for Streamlit playback."""
        text_answer = self.answer(user_input)

        # Convert text to speech
        tts = gTTS(text=text_answer, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        return text_answer, fp.read()
# -------------------------------
# Interview Agent Node
# -------------------------------
class InterviewAgent:
    def __init__(self, llm: GeminiLLM, jd_text: str):
        self.llm = llm
        self.job_description = jd_text
        self.scores = {"intro": 0, "technical": 0, "project": 0}
        self.transcript = []

    def _ask_and_evaluate(self, question: str, section: str, max_score: int, user_answer: str):
        self.transcript.append({"question": question, "answer": user_answer})
        eval_prompt = f"""
You are evaluating an interview response.
Question: {question}
Candidate Answer: {user_answer}
Scoring (0–{max_score}):
- Assess correctness, clarity, detail, relevance.
- Output only JSON: {{"score": <number>, "feedback": "<short feedback>"}}
"""
        eval_response = self.llm.generate(eval_prompt)
        try:
            parsed = json.loads(eval_response)
            self.scores[section] += parsed.get("score", 0)
            return parsed
        except:
            return {"score": 0, "feedback": "Evaluation parse failed."}

    def generate_tech_question(self, i: int):
        tech_prompt = f"Generate a single concise technical interview question suitable for this JD:\n{self.job_description}"
        return self.llm.generate(f"Technical question {i}: {tech_prompt}")

# -------------------------------
# Hybrid Retrieval
# -------------------------------
def hybrid_retrieval_bm25(vectorstore, chunks, query, k: int = 5, alpha: float = 0.5):
    semantic_docs = vectorstore.similarity_search_with_score(query, k=len(chunks))
    semantic_texts = [doc.page_content for doc, _ in semantic_docs]
    semantic_scores = np.array([score for _, score in semantic_docs])
    semantic_scores = 1 / (1 + semantic_scores)  # distance -> similarity

    tokenized_corpus = [doc.split() for doc in semantic_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(query.split()))
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    final_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
    ranked_indices = np.argsort(final_scores)[::-1][:k]
    return [(semantic_texts[i], final_scores[i]) for i in ranked_indices]

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="ChickoHR", layout="wide")
st.title("ChickoHR")

# Sidebar: Load FAISS index
with st.sidebar:
    st.header("Vector Store")
    default_path = st.text_input("FAISS directory path", value="company_policy_faiss")
    build_embeddings = st.checkbox("Create embeddings on first run (if missing)", value=False)
    st.caption("Ensure the FAISS folder is in the repo or created during runtime.")

# Initialize LLM
llm = GeminiLLM()

# Load or build vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs={"device": "cpu"})
vectorstore = None
chunks = []

if os.path.exists(default_path):
    try:
        vectorstore = FAISS.load_local(default_path, embeddings, allow_dangerous_deserialization=True)
        chunks = [doc.page_content for doc in vectorstore.docstore._dict.values()]
        st.sidebar.success("FAISS index loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load FAISS: {e}")
else:
    if build_embeddings:
        st.sidebar.warning("No builder implemented here. Provide your own build step.")
    else:
        st.sidebar.info("Provide a valid FAISS directory.")



# Mode selection
mode = st.selectbox("Choose a mode", ["Policy Query", "Resume Scoring", "Onboarding", "Interview"])

# -------------------------------
# Policy Query UI
# -------------------------------
if mode == "Policy Query":
    st.subheader("Policy Query")
    query = st.text_input("Ask a policy question:")
    if st.button("Ask") and query:
        if vectorstore is None or len(chunks) == 0:
            st.error("Vector store not loaded.")
        else:
            agent =ConversationalRAGAgent(
    lambda q, k=3, alpha=0.6: hybrid_retrieval_bm25(vectorstore, chunks, q, k=k, alpha=alpha),
    llm
)
            answer, docs = agent.ask(query)
            st.write("Answer:")
            st.write(answer)
            with st.expander("Retrieved context"):
                for d in docs:
                    st.markdown("---")
                    st.write(d)

# -------------------------------
# Resume Scoring UI
# -------------------------------
elif mode == "Resume Scoring":
    st.subheader("Resume Screening and Scoring (2 stages)")

    jd = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    resumes = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

    if "qualified_resumes" not in st.session_state:
        st.session_state.qualified_resumes = None
    if "scored_resumes" not in st.session_state:
        st.session_state.scored_resumes = None

    if st.button("Run Screening (Stage 1)"):
        if not jd or not resumes:
            st.warning("Upload the JD and at least one resume PDF first.")
        else:
            with st.spinner("Checking minimum qualifications..."):
                qualified = check_minimum_qualifications(
                    resume_files=resumes,
                    job_desc_file_like=jd,
                    gemini_api_key=st.secrets["GEMINI_API_KEY"],
                )
            st.session_state.qualified_resumes = qualified
            st.session_state.scored_resumes = None
            st.success(f"Qualified candidates: {len(qualified)}")
            st.json(qualified)

    if st.button("Score Qualified (Stage 2)"):
        if not jd:
            st.warning("Upload the JD PDF first.")
        elif not st.session_state.qualified_resumes:
            st.warning("Run Stage 1 first; no qualified candidates available.")
        else:
            with st.spinner("Scoring qualified resumes..."):
                scored = score_qualified_resumes(
                    qualified_resumes=st.session_state.qualified_resumes,
                    job_desc_file_like=jd,
                    gemini_api_key=st.secrets["GEMINI_API_KEY"],
                )
            st.session_state.scored_resumes = scored
            st.success("Scoring complete.")
            st.json(scored)


# -------------------------------
# Onboarding UI
# -------------------------------
# -------------------------------
# Onboarding UI
# -------------------------------
elif mode == "Onboarding":
    st.subheader("Onboarding")
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
    with col2:
        department = st.text_input("Department")
        personal_email = st.text_input("Personal Email")

    if st.button("Generate Welcome + Credentials"):
        if vectorstore is None or len(chunks) == 0:
            st.error("Vector store not loaded.")
        else:
            onboarding_agent = OnboardingAgent(
                llm, vectorstore, chunks, first_name, last_name, department, personal_email,
                retriever_fn=lambda q, k=3, alpha=0.6: hybrid_retrieval_bm25(vectorstore, chunks, q, k=k, alpha=alpha)
            )

            # 🔹 now build_welcome_message returns (text, audio)
            text_msg, audio_bytes = onboarding_agent.build_welcome_message()
            st.write(text_msg)
            st.audio(audio_bytes, format="audio/mp3", start_time=0)

            st.session_state.onboard_agent = onboarding_agent

    ask_q = st.text_input("Ask an onboarding question:")
    if st.button("Ask Onboarding Agent"):
        agent = st.session_state.get("onboard_agent", None)
        if not agent:
            st.error("Create the onboarding agent first.")
        else:
            text_answer, audio_bytes = agent.answer_with_audio(ask_q)
            st.write(text_answer)
            st.audio(audio_bytes, format="audio/mp3", start_time=0)

# -------------------------------
# Interview UI
# -------------------------------
elif mode == "Interview":
    st.subheader("Interview Simulator")
    jd_pdf = st.file_uploader("Upload JD (PDF)", type=["pdf"])
    if jd_pdf:
        jd_text = extract_text_from_pdf(jd_pdf)
        agent = InterviewAgent(llm, jd_text)
        st.session_state.interview_agent = agent
        st.success("JD loaded. Start below.")

    agent = st.session_state.get("interview_agent", None)
    if agent:
        st.markdown("Intro")
        intro_q = "Please introduce yourself."
        intro_ans = st.text_area("Answer to intro:", height=100)
        if st.button("Evaluate Intro"):
            res = agent._ask_and_evaluate(intro_q, "intro", 10, intro_ans)
            st.json(res)

        st.markdown("---")
        st.markdown("Technical Questions")
        for i in range(1, 4):
            if f"tq_{i}" not in st.session_state:
                st.session_state[f"tq_{i}"] = agent.generate_tech_question(i)
            st.write(f"Q{i}: {st.session_state[f'tq_{i}']}")
            tech_ans = st.text_area(f"Answer Q{i}:", key=f"ta_{i}", height=100)
            if st.button(f"Evaluate Q{i}"):
                res = agent._ask_and_evaluate(st.session_state[f"tq_{i}"], "technical", 80 // 3, tech_ans)
                st.json(res)

        st.markdown("---")
        st.markdown("Project")
        proj_ans = st.text_area("Tell me about one project from your resume in detail.", height=150)
        if st.button("Evaluate Project"):
            res = agent._ask_and_evaluate("Tell me about one project from your resume in detail.", "project", 10, proj_ans)
            st.json(res)

        if st.button("Show Final Evaluation"):
            total_score = sum(agent.scores.values())
            result = {"scores": agent.scores, "total_score": total_score, "transcript": agent.transcript}
            st.json(result)
