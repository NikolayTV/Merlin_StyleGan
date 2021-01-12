import os
import sys
import time

from inference import *
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/')
@app.route('/styleganAPI', methods=['POST', 'GET'])
def requests_processor():
    if request.method == 'POST':
        obj_id = request.args.get('id', '')
        changes_type = request.args.get('type', '')
        coeff = request.args.get('coeff', '')
        base64_image = request.args.get('image', '')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8889', threaded=True)