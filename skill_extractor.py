from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_description = """
Data Analyst required with Python, SQL, Excel, Power BI and Pandas.
"""

candidate_skills = """
Python, SQL, Excel, Pandas
"""

documents = [
    job_description,
    candidate_skills
]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:2]
)

print("Cosine Similarity:")

print(similarity[0][0])

print("\nMatch Percentage:")

print(round(similarity[0][0] * 100, 2), "%")