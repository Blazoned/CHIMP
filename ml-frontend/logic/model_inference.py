from os import environ

import numpy as np
import requests


class FacialExpressionModel:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def __init__(self):
        pass

    def predict(self, image):
        # Reshape input, and preprocess pixel to value between 0 and 1
        image = np.array(image).reshape((-1, 48, 48, 1)) / 255

        # Post image to inference server
        url = environ['MODEL_INFERENCE_URL'] + '?stage=staging'     # TODO: Simulate blue-green testing
        headers = {
            'Content-Type': 'application/json'
        }
        json_data = {
            'inputs': image.tolist()
        }
        response = requests.request('POST', headers=headers, url=url, json=json_data)

        # Unpack and return response ordered from most to least likely emotion
        if response.status_code == 200:
            preds = []  # TODO: Test inference via inference server
            class_responses = zip(FacialExpressionModel.EMOTIONS, preds)
            return sorted(class_responses, key=lambda item: item[1], reverse=True)

        # Return empty list if no response was found
        return []
