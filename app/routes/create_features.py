from app import app
from flask import  request, abort, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pickle
import os 
import random
import string
import shutil 

FEATURES_FOLDER = './assets/trackers'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/create-features', methods=['POST'])
def create_features():
    # File Upload Validation
    if 'file' not in request.files:
        abort(400, description="No file part")
    file = request.files['file']
    if file.filename == '':
        abort(400, description="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
    else:
        abort(400, description="Allowed file types are .png, .jpg, .jpeg")

    # Order ID Validation
    if 'order_id' not in request.form:
        abort(400, description="No order id")
    order_id = request.form['order_id']    
        
    # Generate a random 6-character ID
    id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
    # Create the necessary directories
    id_folder = os.path.join(FEATURES_FOLDER, id)
    os.makedirs(id_folder, exist_ok=True)
    preview_folder = os.path.join(id_folder, 'marker-preview')
    os.makedirs(preview_folder, exist_ok=True)
    file_folder = os.path.join(id_folder, 'marker')
    os.makedirs(file_folder, exist_ok=True)
        
    # Read the image directly into a numpy array
    npimg = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
    # Extract the extension from the original filename
    extension = os.path.splitext(filename)[1]

    # Resize the image to 500x500 and save it
    img_resized = cv2.resize(img, (500, 500))
    cv2.imwrite(os.path.join(preview_folder, 'preview-image' + extension), img_resized)
        
    # Convert the image to grayscale for feature extraction
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Compute keypoints and descriptors
    kp, des = orb.detectAndCompute(img_gray, None)

    # Image Feature Extraction Validation
    if des is None or len(des) <= 10:
        abort(400, description="Image does not have enough features.")

    # Convert keypoints to a serializable format
    kp = [kp[i].pt for i in range(len(kp))]

    # Check if the same features already exist
    for tracker_folder in os.listdir(FEATURES_FOLDER):
        marker_file_folder = os.path.join(FEATURES_FOLDER, tracker_folder, 'marker')
        if os.path.isdir(marker_file_folder):  # Check if it's a directory
            for feature_file in os.listdir(marker_file_folder):
                feature_file_path = os.path.join(marker_file_folder, feature_file)
                if os.path.isfile(feature_file_path):  # Check if it's a file, not a directory
                    with open(feature_file_path, 'rb') as f:
                        existing_features = pickle.load(f)
                        if np.array_equal(existing_features['descriptors'], des):
                            shutil.rmtree(id_folder)  # Delete the newly created id folder
                            return jsonify({'message': 'Same features already exist.'})

    # Save the features to a file
    features_path = os.path.join(file_folder, order_id + '.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump({'keypoints': kp, 'descriptors': des}, f)

    return jsonify({
    'marker_preview_path': os.path.join(preview_folder, 'preview-image' + extension),
    'marker_file_path': features_path,
    'tracker_id': id
})
