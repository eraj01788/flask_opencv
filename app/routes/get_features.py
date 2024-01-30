from flask import request, abort, jsonify
import cv2
import numpy as np
from app import app

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/get-features', methods=['POST'])
def get_features():
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

        # Compute the features of the image
        kp, des = orb.detectAndCompute(img, None)
        if des is None:
            features_count = 0
        else:
            features_count = len(des)

        return jsonify({
         'feature_count': features_count, 
      })
    else:
        abort(400, description="Allowed file types are .png, .jpg, .jpeg")