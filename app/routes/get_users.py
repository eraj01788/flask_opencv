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

@app.route('/get-user', methods=['POST'])
def get_users():
    if 'file' not in request.files:
        return 'No image file found', 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        image = Image.open(file)
        image_np = np.array(image)
        
        face_encodings = encode_faces(image_np)

        if not face_encodings:
            response = {'message': 'No face found', 'email': 'none'}
            return json.dumps(response), 200

        # Iterate over all JSON files in the profile folder
        for filename in os.listdir(PROFILE_FOLDER):
            if filename.endswith('.json'):
                with open(os.path.join(PROFILE_FOLDER, filename), 'r') as f:
                    stored_encodings = json.load(f)

                # Compare the uploaded face encodings to the stored encodings
                for stored_encoding in stored_encodings:
                    matches = face_recognition.compare_faces([np.array(stored_encoding)], np.array(face_encodings[0]))
                    if True in matches:
                        profile_name = os.path.splitext(filename)[0]
                        response = {'email': profile_name}
                        return json.dumps(response), 200

        return 'No match found', 200
    else:
        return 'Invalid file type', 400

def encode_faces(image_np):
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    return [encoding.tolist() for encoding in face_encodings]