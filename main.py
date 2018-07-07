import os
from flask import Flask, request, abort, jsonify
import face_recognition
import json
from werkzeug.utils import secure_filename
import numpy as np


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.debug = True

face_encodes = []
data_dir = "./data"

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/v1/post_face_auth', methods=['POST'])
def post_face_auth():

    if "file" not in request.files:
        return json.dumps({"code": "1", "message": "no file"})

    file = request.files["file"]
    if file.filename == '':
        return json.dumps({"code": "2", "message": "no filename"})

    if not file or not allowed_file(file.filename):
        return json.dumps({"code": "3", "message": "no support image"})

    filename = secure_filename(file.filename)
    filename = os.path.join(data_dir, filename)
    file.save(filename)
    encs = get_face_encoding(filename)
    os.remove(filename)
    if len(encs) == 0:
        return json.dumps({"code": "4", "message": "no face"})

    if len(face_encodes) == 0:
        return json.dumps({"code": "5", "message": "no add face"})
    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance(np.array(face_encodes), encs[0])
    for i, face_distance in enumerate(face_distances):
        if face_distance < 0.5:
            return json.dumps({"code": "0", "message": str(face_distance)})

    return json.dumps({"code": "6", "message": "auth failed"})

@app.route('/v1/add-face-encodings', methods=['POST'])
def post_face_encodings():

    if "file" not in request.files:
        return json.dumps({"code": "1", "message": "no file"})

    file = request.files["file"]
    if file.filename == '':
        return json.dumps({"code": "2", "message": "no filename"})

    if not file or not allowed_file(file.filename):
        return json.dumps({"code": "3", "message": "no support image"})

    filename = secure_filename(file.filename)
    filename = os.path.join(data_dir, filename)
    file.save(filename)
    encs = get_face_encoding(filename)
    os.remove(filename)
    if len(encs) == 0:
        return json.dumps({"code": "4", "message": "no face"})
    for enc in encs:
        face_encodes.append(enc)
    return json.dumps({"code": "0", "message": ""})


def get_face_encoding(file):
    image = face_recognition.load_image_file(file)
    encoding = face_recognition.face_encodings(image)
    return encoding


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.makedirs(data_dir) 
    app.run(debug=True, host='0.0.0.0')


