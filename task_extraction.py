import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from datetime import datetime

nltk.download('punkt', force=True)
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Heuristic-based function to extract tasks
def extract_tasks(text):
    sentences = sent_tokenize(text)  # Split into sentences
    task_list = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        named_entities = ne_chunk(tagged_words, binary=False)
        
        # Identify actionable verbs (e.g., "buy", "clean", "submit")
        action_verbs = [word for word, tag in tagged_words if tag.startswith("VB")]
        
        # Check for assignment (e.g., "John has to buy groceries")
        if "has to" in sentence or any(verb in sentence.lower() for verb in ["should", "must", "need to", "is required to"]):
            
            # Extract the responsible person (Named Entity or Subject)
            responsible = next((word for word, tag in tagged_words if tag.startswith("NNP")), "Unknown")
            
            # Extract deadline (if any date/time is present)
            deadline_match = re.search(r'\b(by|before|at)\s+(\d{1,2}\s*(AM|PM|am|pm)?|today|tomorrow|next week)\b', sentence, re.IGNORECASE)
            deadline = deadline_match.group(2) if deadline_match else "No deadline"
            
            task_list.append({
                "task": sentence,
                "action": action_verbs,
                "responsible": responsible,
                "deadline": deadline
            })
    
    return task_list

# Example usage
sample_text = '''Rahul has to clean the room by 5 PM. 
                John should submit the report tomorrow. 
                Alice is outside now.'''
extracted_tasks = extract_tasks(sample_text)

for task in extracted_tasks:
    print(f"Task: {task['task']}")
    print(f"Action: {task['action']}")
    print(f"Responsible: {task['responsible']}")
    print(f"Deadline: {task['deadline']}")
    print("-")
