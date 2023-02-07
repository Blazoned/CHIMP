from os import environ
import logging

from flask_socketio import SocketIO, emit
from flask import request

from logic.image_processor import ImageProcessor


_logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))


def _on_connect():
    _logger.debug(f'Web client connected: {request.sid}')


def _on_disconnect():
    _logger.debug(f'Web client disconnected: {request.sid}')


def _process_image(blob):
    # TODO: cache an image processor per client sid (use connect & disconnect)
    # TODO: only trigger inferences once per second (or 5 seconds), then reuse same results (no lag)
    img_processor = ImageProcessor().load_image(blob)
    img_processor.process()

    emit('update-data', img_processor.predictions)

    return img_processor.get_image_blob()


def add_as_websocket_handler(socket_io: SocketIO):
    global _on_connect, _on_disconnect, _process_image

    _on_connect = socket_io.on('connect')(_on_connect)
    _on_disconnect = socket_io.on('disconnect')(_on_disconnect)
    _process_image = socket_io.on('process-image')(_process_image)

    return socket_io
