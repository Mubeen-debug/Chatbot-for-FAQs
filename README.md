# Chatbot-for-FAQs
pip install nltk spacy flask
python -m spacy download en_core_web_sm
faqs = {
    "What is the return policy?": "Our return policy allows returns within 30 days of purchase.",
    "What are your hours of operation?": "We are open from 9 AM to 6 PM, Monday through Friday.",
    "Where is your store located?": "Our store is located at 123 Main Street, Springfield.",
    "How can I contact customer support?": "You can contact our customer support at support@example.com or call us at (555) 123-4567."
}
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_keys = list(faqs.keys())
vectorizer = TfidfVectorizer().fit_transform(faq_keys)

def find_best_match(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectorizer).flatten()
    best_match_index = similarities.argmax()
    return faq_keys[best_match_index], similarities[best_match_index]
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    processed_input = " ".join(preprocess(user_input))
    best_match, confidence = find_best_match(processed_input)
    
    if confidence > 0.5:  # Threshold for similarity
        response = faqs[best_match]
    else:
        response = "I'm sorry, I couldn't find an answer to your question."
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
