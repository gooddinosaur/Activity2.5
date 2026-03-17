import os
import requests
from flask import Flask, render_template, request, jsonify
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    """Generates a caption for the given image using local Hugging Face model."""
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(images=raw_image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": caption}]
    except Exception as e:
        return {"error": str(e)}

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

    api_response = generate_caption(filepath)
    
    caption = "Caption generation failed."
    if isinstance(api_response, list) and len(api_response) > 0:
        caption = api_response[0].get('generated_text', caption)
    elif "error" in api_response:
        caption = f"API Error: {api_response['error']}"

    return jsonify({
        'image_url': filepath,
        'caption': caption
    })

if __name__ == '__main__':
    app.run(debug=True)
