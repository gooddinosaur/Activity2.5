import os
import requests
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Replace with your actual Hugging Face API token
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": "Bearer YOUR_HUGGING_FACE_TOKEN_HERE"}

def query_huggingface_api(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(HF_API_URL, headers=headers, data=data)
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Call Hugging Face API
    api_response = query_huggingface_api(filepath)
    
    caption = "Caption generation failed."
    if isinstance(api_response, list) and len(api_response) > 0:
        caption = api_response[0].get('generated_text', caption)

    return jsonify({
        'image_url': filepath,
        'caption': caption
    })

if __name__ == '__main__':
    app.run(debug=True)
    