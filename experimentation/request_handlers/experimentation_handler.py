from flask import Response


def _train_model():
    return Response(response='Pong!', status=200)


def _calibrate_model():
    return Response(response='Hello world!', status=200)


def add_as_route_handler(app):
    global _train_model, _calibrate_model

    _train_model = app.route('/model/train', methods=['PUT'])(_train_model)
    _calibrate_model = app.route('/model/calibrated', methods=['PATCH'])(_calibrate_model)

    return app


def add_as_websocket_handler(app):
    global _calibrate_model

    _calibrate_model = app.route('/model/calibrated', methods=['PATCH'])(_calibrate_model)

    return app
