# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# from PIL import Image
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

# # Load the trained model
# model = load_model('leaf_model.h5')

# # Class names (update if different)
# CLASS_NAMES = ['healthy', 'disease1', 'disease2', 'disease3']

# # Ensure uploads folder exists
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))  # Match training size
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     return img_array

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("✅ /predict hit")
#     # if request.method == 'GET':
#     #     return "GET works!"

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("✅ /predict hit")

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']

#     if file.filename == '' or not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     try:
#         img = preprocess_image(file_path)
#         prediction = model.predict(img)
#         predicted_index = np.argmax(prediction)
#         predicted_label = CLASS_NAMES[predicted_index]
#         confidence = float(np.max(prediction))

#         return jsonify({
#             'class': predicted_label,
#             'confidence': round(confidence, 4)
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     print("Available routes:")
#     print(app.url_map)
#     app.run(debug=True)




from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

# Load the trained model
model = load_model('leaf_model_finetuned.h5')
CLASS_NAMES = ['early_blight', 'target_spot', 'curl_virus', 'healthy']
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    # return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].upper() in {'JPG', 'JPEG', 'PNG'}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("✅ /predict hit")

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']

#     if file.filename == '' or not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type'}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     try:
#         img = preprocess_image(file_path)
#         prediction = model.predict(img)
#         predicted_index = np.argmax(prediction)
#         predicted_label = CLASS_NAMES[predicted_index]
#         confidence = float(np.max(prediction))
#         return jsonify({
#             'class': predicted_label,
#             'confidence': round(confidence, 4)
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(prediction))

        return jsonify({
            'class': predicted_label,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)