import os

from flask import Flask, abort
from request_handlers import health_handler, inference_handler
import logging

app = Flask(__name__)
app = health_handler.add_as_route_handler(app)
app = inference_handler.add_as_route_handler(app)

logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('socketio').setLevel(logging.INFO)
logging.getLogger('engineio').setLevel(logging.INFO)


@app.route('/')
def index():
    return abort(418)


def main():
    # Currently functioning as middleware to invoke inference using the mlflow serving api to load a models from
    #   endpoint defined in environment variables.

    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'
    os.environ['MLFLOW_TRACKING_URI'] = 'http://blazoned.nl:8999'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://blazoned.nl:9000'
    os.environ['MODEL_NAME'] = 'onnx emotion model'

    app.run(host='0.0.0.0', port=8500)


if __name__ == '__main__':
    main()
