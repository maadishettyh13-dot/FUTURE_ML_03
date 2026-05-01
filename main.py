
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Resume/Resume.csv")

print("Dataset Columns:", df.columns)

# Correct column name
text_column = 'Resume_str'

# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df[text_column].apply(clean_text)

# -----------------------------
# 3. JOB DESCRIPTION
# -----------------------------
job_desc = """
Looking for a Python Developer with skills in Python, SQL, and data analysis.
"""

job_desc_clean = clean_text(job_desc)

# -----------------------------
# 4. SKILL EXTRACTION
# -----------------------------
skills_list = [
    "python", "machine learning", "data analysis",
    "sql", "nlp", "tensorflow", "pandas", "numpy"
]

def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

df['skills'] = df['cleaned'].apply(extract_skills)
job_skills = extract_skills(job_desc_clean)

# -----------------------------
# 5. TF-IDF + SIMILARITY
# -----------------------------
texts = list(df['cleaned']) + [job_desc_clean]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)

job_vector = vectors[-1]
resume_vectors = vectors[:-1]

df['score'] = cosine_similarity(resume_vectors, job_vector)

# Round scores for better readability
df['score'] = df['score'].round(3)

# -----------------------------
# 6. SKILL GAP
# -----------------------------
def skill_gap(candidate, job):
    return list(set(job) - set(candidate))

df['missing_skills'] = df['skills'].apply(
    lambda x: skill_gap(x, job_skills)
)

# -----------------------------
# 7. RANKING
# -----------------------------
ranked = df.sort_values(by='score', ascending=False)

# Short resume text for display
def short_text(text):
    return text[:150] + "..."

ranked['short_resume'] = ranked[text_column].apply(short_text)

# Get top 5 candidates
top5 = ranked.head(5)

# -----------------------------
# 8. OUTPUT
# -----------------------------
print("\nTop 5 Candidates:\n")
print(top5[['short_resume', 'score', 'skills', 'missing_skills']])

# -----------------------------
# 9. EXPLANATION
# -----------------------------
print("\nExplanation:\n")
print("Candidates are ranked based on similarity between resume and job description.")
print("Higher score = better match.")
print("Missing skills show gaps compared to job requirements.")

# -----------------------------
# 10. VISUALIZATION (BONUS)
# -----------------------------
plt.figure()
plt.barh(range(len(top5)), top5['score'])
plt.yticks(range(len(top5)), [f"Candidate {i+1}" for i in range(len(top5))])
plt.xlabel("Similarity Score")
plt.title("Top 5 Candidate Ranking")
plt.show()