from app import app
from flask import request
import os
import numpy as np
import json
from PIL import Image
import face_recognition

PROFILE_FOLDER = './assets/profiles/json_file'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/create-user', methods=['POST'])
def create_user():
    if 'file' not in request.files:
        return 'No image file found', 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        if not os.path.exists(PROFILE_FOLDER):
            os.makedirs(PROFILE_FOLDER)
        
        image = Image.open(file)
        image_np = np.array(image)
        
        face_encodings = encode_faces(image_np)

        json_file_path = os.path.join(PROFILE_FOLDER, file.filename.rsplit('.', 1)[0] + '.json')
        with open(json_file_path, 'w') as f:
            json.dump(face_encodings, f)
        
        print(f"JSON file saved at: {json_file_path}")  # Debug print

        return 'User Profile Saved', 200
    else:
        return 'Invalid file type', 400

def encode_faces(image_np):
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    return [encoding.tolist() for encoding in face_encodings]