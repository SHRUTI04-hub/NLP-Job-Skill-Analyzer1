# NLP-Based Job Skill Extractor & Skill Gap Analyzer

## 📌 About the Project

An NLP-based web application that analyzes a Job Description and a candidate's Resume to identify required skills, matched skills, missing skills, and resume-job similarity.

## 🛠️ Technologies Used

- Python
- NLP
- NLTK
- Scikit-learn
- TF-IDF
- Cosine Similarity
- pypdf
- Streamlit

## ✨ Features

- Resume PDF Upload
- Job Description Analysis
- NLP Text Preprocessing
- Tokenization
- Stopword Removal
- Lemmatization
- Skill Extraction
- Matched Skills Detection
- Missing Skills Detection
- Resume-Job Similarity Score
- Skill Gap Recommendations
- Streamlit Web Interface

## 🔄 Project Workflow

Job Description + Resume PDF  
↓  
Text Extraction  
↓  
NLP Preprocessing  
↓  
Skill Extraction  
↓  
Skill Matching  
↓  
TF-IDF Vectorization  
↓  
Cosine Similarity  
↓  
Skill Gap Analysis  
↓  
Recommendations

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
