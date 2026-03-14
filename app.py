import streamlit as st
import PyPDF2
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ─── Page Config ───
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ─── Title ───
st.title("AI Resume Analyzer & Job Match Scorer")
st.markdown("Upload your resume and paste a job description to get an **ML match score** + **AI-powered suggestions**.")
st.divider()

# ─── Helper Functions ───

def extract_text_from_pdf(uploaded_file):
    """Extract all text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text.strip()

def get_ml_score(resume_text, jd_text):
    """Use TF-IDF + Cosine Similarity to score resume vs job description."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def get_ai_feedback(resume_text, jd_text):
    """Call Groq API to get detailed career coaching feedback."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = f"""
You are an expert HR recruiter and career coach with 10+ years of experience.

RESUME:
{resume_text[:2500]}

JOB DESCRIPTION:
{jd_text[:2500]}

Please provide a structured analysis with exactly these 4 sections:

**1. Top 3 Matching Strengths**
(What the candidate already has that fits this job)

**2. Top 3 Skill Gaps**
(What is missing or weak compared to the job requirements)

**3. 3 Resume Bullet Points to Add**
(Specific, copy-paste ready bullet points the candidate should add)

**4. Overall Verdict**
(One of: Strong Match / Moderate Match / Weak Match — and one sentence why)

Be specific, honest, and actionable. Avoid generic advice.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content

# ─── API Key Input ───
st.subheader("Enter Your GROQ API Key")
api_key = st.text_input(
    "API Key",
    type="password",
    placeholder="gsk_...",
    help="Get your free key at console.groq.com"
)

if api_key:
    import os
    os.environ["GROQ_API_KEY"] = api_key

st.divider()

# ─── Input Section ───
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Your Resume")
    resume_file = st.file_uploader(
        "Upload PDF only",
        type=["pdf"],
        help="Your resume in PDF format"
    )
    if resume_file:
        st.success(f"Uploaded: {resume_file.name}")

with col2:
    st.subheader("Paste Job Description")
    jd_text = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Machine Learning intern with Python skills, experience in Scikit-learn, and knowledge of NLP..."
    )
    if jd_text:
        st.success(f"{len(jd_text.split())} words entered")

st.divider()

# ─── Analyze Button ────────────────────────────────────────────
analyze_btn = st.button("Analyze My Resume", type="primary", use_container_width=True)

if analyze_btn:
    # Validation
    if not api_key:
        st.error("Please enter your Anthropic API key above.")
    elif not resume_file:
        st.error("Please upload your resume PDF.")
    elif not jd_text.strip():
        st.error("Please paste the job description.")
    else:
        # ── Extract Text ──
        with st.spinner("Reading your resume..."):
            resume_text = extract_text_from_pdf(resume_file)

        if not resume_text:
            st.error("Could not read text from your PDF. Make sure it's not a scanned image.")
        else:
            # ── ML Score ──
            with st.spinner("Calculating ML match score..."):
                ml_score = get_ml_score(resume_text, jd_text)

            # ── Display Score ──
            st.subheader("Your Match Score")
            
            col_score, col_info = st.columns([1, 2])
            with col_score:
                if ml_score >= 60:
                    st.metric("Match Score", f"{ml_score}%", "🟢 Strong")
                elif ml_score >= 35:
                    st.metric("Match Score", f"{ml_score}%", "🟡 Moderate")
                else:
                    st.metric("Match Score", f"{ml_score}%", "🔴 Needs Work")
            
            with col_info:
                st.progress(ml_score / 100)
                st.caption("Calculated using TF-IDF vectorization + Cosine Similarity (ML)")

            st.divider()

            # ── AI Feedback ──
            with st.spinner("Getting AI feedback from Claude... (this takes ~10 seconds)"):
                try:
                    feedback = get_ai_feedback(resume_text, jd_text)
                    st.subheader("AI-Powered Career Coach Feedback")
                    st.markdown(feedback)
                except Exception as e:
                    st.error(f"AI feedback failed: {str(e)}")
                    st.info("Check that your API key is correct and has credits.")

            st.divider()
            st.success("Analysis complete! Use the feedback above to improve your resume.")