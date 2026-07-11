from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# 1. Load Model and Tokenizer
MODEL_PATH = './bert_stress_model_v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() 

# 2. Route for the Landing Page
@app.route('/')
def landing():
    return render_template('landing.html')

# 3. Route for the Predictor Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

# 4. API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        
        stress_prob = probs[0][1].item()

# Three-level stress classification
        if stress_prob < 0.25:
            stress_level = "Low Stress"
            confidence = (1 - stress_prob)

        elif stress_prob < 0.60:
            stress_level = "Mild Stress"
            confidence = stress_prob

        else:
            stress_level = "High Stress"
            confidence = stress_prob

        print("Stress Probability:", stress_prob)
        print("Prediction:", stress_level)

        result = f"{stress_level} ({confidence * 100:.2f}% confidence)"
        

        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Running in debug mode helps see errors in the terminal
    app.run(debug=True)