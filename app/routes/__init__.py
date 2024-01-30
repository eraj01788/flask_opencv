import os
from flask import send_from_directory
from app import app
from .create_features import create_features
from .find_order import find_order
from .delete_features import delete_features
from .create_user import create_user
from .get_users import get_users
from .get_features import get_features

@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory(os.path.join(os.getcwd(), 'assets'), path)