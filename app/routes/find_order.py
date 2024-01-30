from app import app
from flask import request, abort, jsonify
import os
import cv2
import numpy as np
import pickle


MARKER_FOLDER = './assets/trackers'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/find-order', methods=['POST'])
def find_order():
    if 'file' not in request.files:
        abort(400, description="No file part")
    file = request.files['file']
    if file.filename == '':
        abort(400, description="No selected file")
    if file and allowed_file(file.filename):
        # Read the image directly into a numpy array
        npimg = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

        # Initialize the ORB detector
        orb = cv2.ORB_create()

        # Compute the features of the uploaded image
        kp1, des1 = orb.detectAndCompute(img, None)
        if des1 is None:
            features_count_uploaded = 0
        else:
            features_count_uploaded = len(des1)

        best_match = "Not found"
        max_good_matches = 0
        marker_feature_count = 0

        # Iterate over the folders in the marker folder
        for folder in os.listdir(MARKER_FOLDER):
            marker_file_folder = os.path.join(MARKER_FOLDER, folder, 'marker')
            if os.path.isdir(marker_file_folder):  # Check if it's a directory
                for feature_file in os.listdir(marker_file_folder):
                    feature_file_path = os.path.join(marker_file_folder, feature_file)
                    if os.path.isfile(feature_file_path):  # Check if it's a file, not a directory
                        with open(feature_file_path, 'rb') as f:
                            features = pickle.load(f)
                            kp2 = features['keypoints']
                            des2 = features['descriptors']

                            # Compute matches
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                            matches = bf.match(des1, des2)
                            # Filter matches based on distance
                            distance_threshold = 30  # Set your threshold here
                            matches = [m for m in matches if m.distance < distance_threshold]

                            matches = sorted(matches, key = lambda x:x.distance)

                            good_matches = len(matches)
                            if good_matches > 10 and good_matches > max_good_matches:
                                max_good_matches = good_matches
                                best_match, _ = os.path.splitext(feature_file)  
                                marker_feature_count = len(kp2)

        return jsonify({
         'best_match': best_match, 
         'good_matches': max_good_matches, 
         'uploaded_feature_count': features_count_uploaded, 
         'marker_feature_count': marker_feature_count
      })
    else:
        abort(400, description="Allowed file types are .png, .jpg, .jpeg")