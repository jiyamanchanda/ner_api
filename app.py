from flask import Flask, request, jsonify
import spacy

# Load your trained model
nlp = spacy.load("custom_ner_model")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    doc = nlp(text)

    results = []
    for ent in doc.ents:
        results.append({
            "text": ent.text,
            "label": ent.label_
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

app = Flask(__name__)

@app.route('/')
def home():
    return "NER API is running!"
