import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session

import numpy as np


config = tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)


class FacialExpressionModel:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def __init__(self, model_network_file: str, model_weights_file: str):
        with open(model_network_file, 'r') as f:
            json_model = f.read()
            self._model = model_from_json(json_model)
            self._model.load_weights(model_weights_file)

    def predict(self, img):
        global session
        set_session(session)

        preds = self._model.predict(img)
        results = sorted(zip(FacialExpressionModel.EMOTIONS, np.array(preds).tolist()[0]), key=lambda item: item[1], reverse=True)
        return results
