from os import environ
import logging

from flask_socketio import SocketIO, send, emit
from flask import Flask, render_template, request


_logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))

def _on_connect():
    _logger.debug(f'Web client connected: {request.sid}')


def _on_disconnect():
    _logger.debug(f'Web client disconnected: {request.sid}')


def _process_image(blob):
    return blob


def add_as_websocket_handler(socket_io: SocketIO):
    global _on_connect, _on_disconnect, _process_image

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_image = socket_io.on('process-image')(_process_image)

    return _on_connect
