import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Streamlit UI Configuration
st.set_page_config(page_title="AI Resume Screening", page_icon="🚀", layout="wide")

# Inject Custom CSS for Styling
st.markdown("""
    <style>
        /* Background Styling */
        body {
            background-color: #0a192f;
        }
            
            /* Glowing Header */
        .header {
            background: linear-gradient(135deg, #09131b, #164863);
            padding: 12px;
            #text-align: center;
            border-radius: 8px;
            color: white;
            font-size: 34px;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(0, 128, 255, 0.8);
            margin-bottom: 25px;
        }

        /* Glowing Border Effect */
        @keyframes glow {
            0% { border-color: #38b6ff; box-shadow: 0 0 5px #38b6ff; }
            50% { border-color: #00d4ff; box-shadow: 0 0 15px #00d4ff; }
            100% { border-color: #38b6ff; box-shadow: 0 0 5px #38b6ff; }
        }

        /* Main Container */
        .main-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            padding: 20px;
            border-radius: 15px;
            border: 3px solid transparent;
        }

/* Input Field */
        .stTextArea textarea, .stTextInput input, y{
            border: 2px solid;
            animation: glow 2s infinite alternate;
            color: white;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #09131b, #164863);
            color: white;
            padding: 20px;
        }

        /* Custom Button */
        .stButton>button {
            background: linear-gradient(135deg, #09131b, #164863);
            color: white;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            border: none;
            transition: 0.3s;
            animation: glow 2s infinite alternate;
        }

        /* Title Styling */
        h1 {
            color: #38b6ff;
            #text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        }
            
            /* Footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(135deg, #09131b, #164863);
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }

        .footer a {
            color: #38b6ff;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

             .logo-container {
            display: flex;
            align-items: center;
            gap: 10px; /* Reduce space between image and text */
        }
        .logo-container img {
            border-radius: 50%; /* Makes image circular */
            width: 60px; /* Adjust size */
            height: 60px;
            object-fit: cover; /* Ensures the image fits well */
        }
        .logo-container h1 {
            margin: 0;
            color: #38b6ff;
        }
            
    </style>
""", unsafe_allow_html=True)

# Main UI Layout
col1, col2 = st.columns([1, 7])  # Adjust column width for alignment

with col1:
    st.image("CURLS.png", width=80)  # Load local image

with col2:
    st.markdown("<h1 class='header' style='padding-left:30px'>ResumeVista-<br> AI Resume Screening & Ranking System</h1></div>", unsafe_allow_html=True)
    st.markdown("---")  # Adds a horizontal line

# Upload Section (Now in Main Content)
with st.container():
    st.markdown("""
        <div style="padding: 15px; border-radius: 10px;">
        <h3 style="color: grey">📂 Upload Resumes</h3>
        </div>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

st.markdown("---")  # Adds a horizontal line

# Job description input
with st.container():
    st.markdown("""
        <div style=" padding-top: 15px; border-radius: 10px;">
        <h3 style="color: grey">📝 Job Description</h3>
        </div>
    """, unsafe_allow_html=True)
    job_description = st.text_area("Enter the job description")

st.markdown("---")  # Adds a horizontal line

if uploaded_files and job_description:
    st.subheader("🔍 Ranking Resumes")
    
    resumes = []
    with st.spinner("Processing resumes..."):
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes.append(text)

    scores = rank_resumes(job_description, resumes)
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)

    # Display styled dataframe
    st.dataframe(results.style.background_gradient(cmap="Blues"))

    # Visualization - Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(results["Resume"], results["Score"], color=["#38b6ff", "#00d4ff", "#006494", "#58a6ff"])
    ax.set_xlabel("Matching Score", fontsize=12, color="white")
    ax.set_ylabel("Resume File", fontsize=12, color="white")
    ax.set_title("📊 Resume Ranking Based on Job Description", fontsize=14, color="white")
    ax.set_facecolor("#0a192f")  # Dark Background
    ax.spines['bottom'].set_color("white")
    ax.spines['left'].set_color("white")
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)

# Sidebar Features
st.sidebar.header("📊 Resume Analysis Summary")
st.sidebar.markdown(f"**Uploaded Files:** {len(uploaded_files) if uploaded_files else 0}")

if uploaded_files:
    total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)  # Convert to MB
    st.sidebar.markdown(f"**Total Size:** {total_size:.2f} MB")
    st.sidebar.markdown("**Recent Uploads:**")
    for file in uploaded_files:
        st.sidebar.text(f"📄 {file.name}")

st.sidebar.header("⚙️ Settings")
min_score = st.sidebar.slider("Minimum Score Threshold", 0.0, 1.0, 0.5, 0.01)
highlight_best = st.sidebar.checkbox("Highlight Best Matching Resume")

# Display Highlighted Best Resume
if highlight_best and uploaded_files and job_description:
    best_resume = results.iloc[0]
    st.sidebar.success(f"📌 Best Match: {best_resume['Resume']} (Score: {best_resume['Score']:.2f})")

st.markdown(f"""
    <div class='footer'>
        &copy; AI Resume Screening and Ranking System | Shreya Chaturvedi
    </div>
""", unsafe_allow_html=True)