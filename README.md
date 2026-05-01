 📄 Resume Screening & Candidate Ranking System (ML + NLP)

🚀 Project Overview

This project is a **Machine Learning–based Resume Screening System** that automatically analyzes, scores, and ranks candidates based on their relevance to a given job description.
The goal is to simulate how modern **HR-tech platforms** and recruitment tools shortlist candidates efficiently using **Natural Language Processing (NLP)**.

 🎯 Objective
 
To build a system that can:
* Extract meaningful information from resumes
* Match resumes with job descriptions
* Rank candidates based on job relevance
* Identify missing or required skills

🧠 How It Works

 1. Text Preprocessing
* Converts resume text to lowercase
* Removes special characters and stopwords
* Prepares clean text for analysis

 2. Skill Extraction
* Uses a predefined skill list
* Extracts relevant skills from resumes and job description

3. Feature Engineering
* Applies **TF-IDF (Term Frequency–Inverse Document Frequency)**
* Converts text into numerical vectors

4. Similarity Scoring
* Uses **Cosine Similarity** to measure match between:
  * Resume
  * Job description

5. Candidate Ranking
* Candidates are ranked based on similarity score
* Higher score = better match

6. Skill Gap Analysis
* Identifies missing skills required for the job
* Helps in decision-making and candidate evaluation

🛠️ Tech Stack
* Python
* Pandas
* NLTK
* Scikit-learn
* Matplotlib

 📊 Features
✔ Resume text preprocessing
✔ Skill extraction using NLP
✔ Resume-job similarity scoring
✔ Candidate ranking system
✔ Skill gap identification
✔ Visualization of top candidates

🧪 Testing & Validation
* Tested with multiple job descriptions (ML, Data Analyst, Web Developer)
* Verified that relevant candidates rank higher
* Skill gap analysis correctly identifies missing skills
* System dynamically adapts to different job role

⚠️ Limitations
* Uses a predefined skill list (can be expanded)
* Basic NLP approach may capture generic keyword overlaps
* Does not yet use deep learning or advanced embeddings

🔮 Future Improvements
* Use advanced NLP models (BERT, spaCy pipelines)
* Improve skill extraction using Named Entity Recognition
* Add web interface for real-time resume upload
* Implement weighted skill importance

 📌 Conclusion
This project demonstrates how Machine Learning and NLP can be used to **automate resume screening**, reduce manual effort, and improve hiring decisions.
It serves as a strong foundation for building real-world **AI-powered recruitment systems**.

 🙌 Acknowledgment
Developed as part of the **Machine Learning Internship Program by Future Interns**.





