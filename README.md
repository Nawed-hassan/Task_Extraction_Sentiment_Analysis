Sentiment Analysis & Task Extraction Project

📌 Overview

This project consists of two parts:

Task Extraction from Unstructured Text (Part A)

Extracts tasks from text using heuristic-based NLP.

Identifies actions, responsible persons, and deadlines.

Sentiment Analysis of Customer Reviews (Part B)

Classifies reviews as positive or negative using Logistic Regression.

Uses TF-IDF for text vectorization.

📂 Project Structure

/project-folder/
│── part_a_task_extraction.py  # Task Extraction (NLP)
│── part_b_sentiment_analysis.py  # Sentiment Analysis (ML)
│── dataset.csv  # Dataset file
│── requirements.txt  # Required Python Libraries
│── README.md  # Project Documentation

⚙️ Setup Instructions

1️⃣ Create a Virtual Environment

Run the following command to create and activate a virtual environment:

python -m venv venv  # Create virtual environment
# Activate it:
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

2️⃣ Install Required Libraries

pip install -r requirements.txt

🚀 Running the Scripts

Part A: Task Extraction

python part_a_task_extraction.py

This script extracts tasks, responsible persons, and deadlines from text.

Part B: Sentiment Analysis

python part_b_sentiment_analysis.py

This script trains and evaluates a sentiment classification model.
