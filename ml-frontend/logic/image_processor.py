from os import path, getcwd
import cv2
import numpy as np

from logic.model_inference import FacialEmotionInference


class ImageProcessor:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    image: np.array
    emotion_inference: FacialEmotionInference = FacialEmotionInference()
    predictions: list

    def __init__(self):
        # Load facial recognition haar cascade
        cascade_file = path.join(getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')

        self.is_processed = True
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def load_image(self, binary_blob: str):
        self.image = cv2.imdecode(np.frombuffer(binary_blob, np.uint8), -1)
        self.predictions = []
        self.is_processed = False

        return self

    def process(self):
        if self.is_processed:
            return

        # Get gray-scale version of the image, detect each face, and get for each face an emotion prediction.
        grey_frame = cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)
        faces = self.face_cascade.detectMultiScale(grey_frame, 1.3, 5)
        for (x, y, width, height) in faces:
            face = cv2.resize(grey_frame[y:y+height, x:x+width], (48, 48))
            preds = self.emotion_inference.predict(face)
            self.predictions.append(preds)

            cv2.putText(self.image, preds[0][0], (x, y-10), self.font, .85, (47, 47, 255), 2)
            cv2.rectangle(self.image, (x, y), (x+width, y+height), (192, 192, 0), 1)

        self.is_processed = True

    def get_image_blob(self) -> str:
        # Get a jpg blob of the image, in string format
        return cv2.imencode('.jpg', self.image)[1].tostring()
