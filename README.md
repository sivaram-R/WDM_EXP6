### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)

# Preprocess the documents
preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Create a TF-IDF vectorizer and fit the documents
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

# Search function to calculate cosine similarity
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity between query and documents
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Sort documents based on similarity scores
    sorted_indexes = similarity_scores.argsort()[0][::-1]

    # Return sorted documents along with their similarity scores
    results = [(list(documents.keys())[i], list(documents.values())[i], similarity_scores[0, i]) for i in sorted_indexes]
    return results

# Input and search results
query = input("Enter your query: ")
search_results = search(query, tfidf_matrix, tfidf_vectorizer)

# Output the results
print("Query:", query)
for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

# Find the highest rank similarity score
highest_rank_score = max(result[2] for result in search_results)
print("The highest rank cosine score is:", highest_rank_score)
```

### Output:
![image](https://github.com/user-attachments/assets/8540de79-4a0a-4ad1-8ad4-89c6b44aa988)

### Result:
Thus the implementation Information Retrieval Using Vector Space Model in Python is successfullly executed.
