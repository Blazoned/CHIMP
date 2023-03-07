from os import path, getcwd, makedirs
from zipfile import ZipFile

from flask import Response, request, abort, jsonify
from werkzeug.utils import secure_filename


# region Training
def _train_model():
    # TODO: CALL TRAINING
    return Response(response='Pong!', status=200)
# endregion


# region Calibration
def _calibrate_model():
    def is_file_allowed(fname: str):
        return '.' in fname and fname.rsplit('.', 1)[1].lower() == 'zip'

    # Check if user id defined
    if 'user_id' not in request.args:
        return abort(400, 'No user specified.')

    # Check if files present in request
    if len(request.files) == 0:
        return abort(400, 'No files uploaded.')

    if 'zipfile' not in request.files:
        return abort(400, 'Different file expected.')

    # Check if file is a valid zip
    file = request.files['zipfile']

    if file.filename == '':
        return abort(400, 'No file selected.')
    if not is_file_allowed(file.filename):
        return abort(400, 'File type not allowed. Must be a zip.')

    # Save zip file
    file_name = secure_filename(file.filename)
    folder_path = path.join(getcwd(), 'uploads', request.args.get('user_id'))
    makedirs(folder_path, exist_ok=True)

    file_path = path.join(folder_path, file_name)
    file.save(file_path)

    # Unpack zip file
    with ZipFile(file_path, 'r') as zipfile:
        zipfile.extractall(folder_path)

    # Call calibration upon folder with the given user id
    # TODO: CALL CALIBRATION WITH FOLDER

    return jsonify(success=True)
# endregion


def add_as_route_handler(app):
    global _train_model, _calibrate_model

    _train_model = app.route('/model/train', methods=['POST'])(_train_model)
    _calibrate_model = app.route('/model/calibrate', methods=['POST'])(_calibrate_model)

    return app
