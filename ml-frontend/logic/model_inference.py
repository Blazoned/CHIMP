from os import environ

from random import random
import numpy as np
import requests
from json import loads


class FacialEmotionInference:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def __init__(self):
        # Simulate blue-green test; proper implementation would have validation for blue-green test
        self.stage = 'production' if random() < .9 else 'staging'

    def predict(self, image):
        # Reshape input, and preprocess pixel to value between 0 and 1
        image = np.array(image).reshape((-1, 48, 48, 1)) / 255

        # Post image to inference server
        url = environ['MODEL_INFERENCE_URL'] + f'?stage={self.stage}'
        headers = {
            'Content-Type': 'application/json'
        }
        json_data = {
            'inputs': image.tolist()
        }
        response = requests.request('POST', headers=headers, url=url, json=json_data)

        # Unpack and return response ordered from most to least likely emotion
        if response.status_code == 200:
            text_response = loads(response.text)
            predictions = list(text_response['predictions'].values())[0][0]  # Unpack response into list of predictions
            class_responses = zip(self.EMOTIONS, predictions)
            return sorted(class_responses, key=lambda item: item[1], reverse=True)

        # Return empty list if no response was found
        return [[]]
