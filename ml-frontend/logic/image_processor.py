from os import path, getcwd
from time import time

import cv2
import numpy as np

from logic.model_inference import FacialEmotionInference


class ImageProcessor:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    image: np.array
    emotion_inference: FacialEmotionInference = FacialEmotionInference()
    predictions: list

    def __init__(self, inference_interval: int):
        self.is_processed = True
        self.inference_interval = inference_interval
        self._previous_inference_call = 0

        # Load facial recognition haar cascade
        cascade_file = path.join(getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def load_image(self, binary_blob: str):
        self.image = cv2.imdecode(np.frombuffer(binary_blob, np.uint8), -1)
        self.is_processed = False

        return self

    def process(self):
        if self.is_processed:
            return

        # Get gray-scale version of the image, detect each face, and get for each face an emotion prediction.
        grey_frame = cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)
        faces = self.face_cascade.detectMultiScale(grey_frame, 1.3, 5)

        # Reset the predictions if new inference call is required
        do_new_inference_call = time() - self._previous_inference_call > self.inference_interval

        if do_new_inference_call:
            self.predictions = []

        for index, (x, y, width, height) in enumerate(faces):
            # Execute inference call if inference call has passed, else use previous results
            if do_new_inference_call:
                face = cv2.resize(grey_frame[y:y+height, x:x+width], (48, 48))
                prediction = self.emotion_inference.predict(face)
                self.predictions.append(prediction)

                # Set inference call time to current time
                self._previous_inference_call = time()

            if index >= len(self.predictions):
                return

            prediction = self.predictions[index]
            cv2.putText(self.image, prediction[0][0], (x, y-10), self.font, .85, (47, 47, 255), 2)
            cv2.rectangle(self.image, (x, y), (x+width, y+height), (192, 192, 0), 1)

        self.is_processed = True

    def get_image_blob(self) -> str:
        # Get a jpg blob of the image, in string format
        return cv2.imencode('.jpg', self.image)[1].tostring()
