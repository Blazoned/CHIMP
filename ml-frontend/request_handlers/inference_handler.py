from flask_socketio import SocketIO, send, emit
from flask import Flask, render_template, request

_app: Flask


def _on_connect():
    _app.logger.debug(f'Web client connected: {request.sid}')


def _on_disconnect():
    _app.logger.debug(f'Web client disconnected: {request.sid}')


def _process_image(blob):
    _app.logger.debug(f'Processing webcam feed for user: {request.sid}')

    return blob


def add_as_websocket_handler(app: Flask, socket_io: SocketIO):
    global _on_connect, _on_disconnect, _process_image
    global _app

    _app = app

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_image = socket_io.on('process-image')(_process_image)

    return _on_connect
