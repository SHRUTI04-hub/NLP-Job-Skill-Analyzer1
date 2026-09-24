import streamlit as st
from pypdf import PdfReader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# NLTK Resources
# -----------------------------------

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Job Skill Analyzer",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("🤖 AI Job Skill Analyzer")

st.write(
    "NLP-based system to analyze job requirements "
    "and identify candidate skill gaps."
)


# -----------------------------------
# NLP Objects
# -----------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------------
# Skills Database
# -----------------------------------

skills = [
    "python",
    "sql",
    "excel",
    "power bi",
    "pandas",
    "numpy",
    "tableau",
    "java",
    "machine learning",
    "deep learning",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "spring boot",
    "mysql",
    "mongodb",
    "data analysis",
    "data visualization",
    "statistics"
]


# -----------------------------------
# NLP Preprocessing Function
# -----------------------------------

def preprocess_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        if token.isalpha() and token not in stop_words:

            lemma = lemmatizer.lemmatize(token)

            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


# -----------------------------------
# Job Description
# -----------------------------------

st.subheader("📌 Job Description")

job_description = st.text_area(
    "Paste the Job Description here:",
    height=200
)


# -----------------------------------
# Resume Upload
# -----------------------------------

st.subheader("📄 Upload Resume")

uploaded_file = st.file_uploader(
    "Upload your Resume PDF",
    type=["pdf"]
)


# -----------------------------------
# Additional Skills
# -----------------------------------

st.subheader("💻 Additional Skills")

candidate_input = st.text_input(
    "Enter additional skills (comma separated):",
    placeholder="Python, SQL, Excel, Power BI"
)


# -----------------------------------
# Analyze Button
# -----------------------------------

if st.button("🔍 Analyze Resume", use_container_width=True):

    if not job_description:

        st.warning(
            "Please enter the Job Description."
        )

    elif not uploaded_file:

        st.warning(
            "Please upload your Resume PDF."
        )

    else:

        # -----------------------------------
        # Extract Resume Text
        # -----------------------------------

        reader = PdfReader(uploaded_file)

        resume_text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                resume_text += page_text + " "


        # -----------------------------------
        # Combine Resume + Additional Skills
        # -----------------------------------

        candidate_text = (
            resume_text + " " + candidate_input
        )


        # -----------------------------------
        # NLP Preprocessing
        # -----------------------------------

        processed_job = preprocess_text(
            job_description
        )

        processed_resume = preprocess_text(
            candidate_text
        )


        # -----------------------------------
        # Skill Extraction
        # -----------------------------------

        job_lower = job_description.lower()

        resume_lower = candidate_text.lower()


        required_skills = []

        candidate_skills = []


        for skill in skills:

            if skill in job_lower:

                required_skills.append(skill)

            if skill in resume_lower:

                candidate_skills.append(skill)


        # -----------------------------------
        # Matched & Missing Skills
        # -----------------------------------

        matched_skills = []

        missing_skills = []


        for skill in required_skills:

            if skill in candidate_skills:

                matched_skills.append(skill)

            else:

                missing_skills.append(skill)


        # -----------------------------------
        # TF-IDF
        # -----------------------------------

        documents = [
            processed_job,
            processed_resume
        ]


        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )


        # -----------------------------------
        # Cosine Similarity
        # -----------------------------------

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )


        similarity_percentage = (
            similarity[0][0] * 100
        )


        # -----------------------------------
        # Results
        # -----------------------------------

        st.success(
            "Analysis completed successfully! ✅"
        )


        # -----------------------------------
        # Similarity Score
        # -----------------------------------

        st.subheader("📊 Resume-Job Similarity")

        st.metric(
            "Similarity Score",
            f"{similarity_percentage:.2f}%"
        )


        # -----------------------------------
        # Required Skills
        # -----------------------------------

        st.subheader("🎯 Required Skills")

        if required_skills:

            st.write(
                ", ".join(required_skills)
            )

        else:

            st.write(
                "No predefined skills detected."
            )


        # -----------------------------------
        # Matched Skills
        # -----------------------------------

        st.subheader("✅ Matched Skills")

        if matched_skills:

            for skill in matched_skills:

                st.success(
                    "✓ " + skill
                )

        else:

            st.write(
                "No matched skills found."
            )


        # -----------------------------------
        # Missing Skills
        # -----------------------------------

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:

                st.error(
                    "✗ " + skill
                )

        else:

            st.success(
                "No missing required skills! 🎉"
            )


        # -----------------------------------
        # Resume Skills
        # -----------------------------------

        st.subheader("💼 Skills Found in Resume")

        if candidate_skills:

            st.write(
                ", ".join(candidate_skills)
            )

        else:

            st.write(
                "No predefined skills detected."
            )


        # -----------------------------------
        # Recommendations
        # -----------------------------------

        st.subheader("🚀 Skill Recommendations")

        if missing_skills:

            for skill in missing_skills:

                st.write(
                    f"→ Consider learning **{skill}**"
                )

        else:

            st.success(
                "Your skills cover all detected job requirements!"
            )


        # -----------------------------------
        # NLP Processed Text
        # -----------------------------------

        with st.expander(
            "🧠 View NLP Preprocessed Text"
        ):

            st.write(
                "Processed Job Description:"
            )

            st.code(processed_job)

            st.write(
                "Processed Resume:"
            )

            st.code(processed_resume)


        # -----------------------------------
        # Extracted Resume Text
        # -----------------------------------

        with st.expander(
            "📄 View Extracted Resume Text"
        ):

            st.write(resume_text)