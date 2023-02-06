from flask_socketio import SocketIO, send, emit
from flask import Flask, render_template, request
import logging


app = Flask(__name__)
socket_io = SocketIO(app, always_connect=True, logger=False, engineio_logger=False)

logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger('socketio').setLevel(logging.INFO)
logging.getLogger('engineio').setLevel(logging.INFO)


@app.route('/')
def index():
    return render_template('index.html')


def get_app():
    return socket_io.run(app=app, host='0.0.0.0', port=5252, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    get_app()
