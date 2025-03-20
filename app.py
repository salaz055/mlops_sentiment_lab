from flask import Flask, request, jsonify, render_template
from analyze import get_sentiment, compute_embeddings, classify_email
import json

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    print("Home page")
    return render_template('index.html')


@app.route("/api/v1/sentiment-analysis/", methods=['POST'])
def analysis():
    if request.is_json:
        data = request.get_json()
        sentiment = get_sentiment(data['text'])
        return jsonify({"message": "Data received", "data": data, "sentiment": sentiment}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/valid-embeddings/", methods=['GET'])
def valid_embeddings():
    embeddings = compute_embeddings()
    formatted_embeddings = []
    for text, vector in embeddings:
        formatted_embeddings.append({
            "text": text,
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
        })
    embeddings = formatted_embeddings
    return jsonify({"message": "Valid embeddings fetched", "embeddings": embeddings}), 200


@app.route("/api/v1/classify/", methods=['POST'])
def classify():
    if request.is_json:
        data = request.get_json()
        text = data['text']
        classifications = classify_email(text)
        return jsonify({"message": "Email classified", "classifications": classifications}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/classify-email/", methods=['GET'])
def classify_with_get():
    text = request.args.get('text')
    classifications = classify_email(text)
    return jsonify({"message": "Email classified", "classifications": classifications}), 200

@app.route('/api/v1/add-class/', methods =['POST'])
def add_class():
    r = request.json
    
    if "Class" not in r:
        return jsonify({'ERROR' : "Please provide a class name ex. Class:Sports "}) , 400
        
    api_class = r['Class']
    
    
    with open('all_classes.json', 'r') as file:
        classes = json.load(file)
        
    classes.append(api_class)
    
    # adding the new class to the all_classes.json file
    with open('all_classes.json', 'w') as file:
        json.dump(classes,file)
    
    return jsonify({"message" : f"Class {api_class} has been added to all_classes.json", "current_classes" : classes}) , 200
    
        

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)