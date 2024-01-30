import os
import shutil
from flask import jsonify
from app import app

@app.route('/delete-features/<string:tracker_id>', methods=['DELETE'])
def delete_features(tracker_id):
    directory = os.path.join(os.getcwd(), 'assets', 'trackers', tracker_id)
    print(directory)
    if os.path.exists(directory):
        shutil.rmtree(directory)
        return jsonify({"message": f"Tracker {tracker_id} deleted successfully"}), 200
    else:
        return jsonify({"message": f"Tracker {tracker_id} not found"}), 404