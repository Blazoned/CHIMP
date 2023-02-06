from os import path, getcwd
import cv2
import numpy as np

# from model_inference import FacialExpressionModel


class ImageProcessor:
    image: np.array
    predictions: list
    # model = FacialExpressionModel('model_data/model.json', 'model_data/model_weights.oversampled.hdf5')

    def __init__(self):
        cascade_file = path.join(getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')

        self.is_processed = True
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def load_image(self, binary_blob: str):
        self.image = cv2.imdecode(np.frombuffer(binary_blob, np.uint8), -1)
        self.predictions = []
        self.is_processed = False

        return self

    def process(self):
        if self.is_processed:
            return

        grey_frame = cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)
        faces = self.face_cascade.detectMultiScale(grey_frame, 1.3, 5)
        for (x, y, width, height) in faces:     # TODO: Enable prediction
            # face = cv2.resize(grey_frame[y:y+height, x:x+width], (48, 48))
            # preds = self.model.predict(face[np.newaxis, :, :, np.newaxis])
            # self.predictions.append(preds)

            # cv2.putText(self._decoded, preds[0][0], (x, y-10), self.font, .85, (47, 47, 255), 2)
            cv2.rectangle(self.image, (x, y), (x+width, y+height), (192, 192, 0), 1)

        self.is_processed = True

    def get_image_blob(self) -> str:
        return cv2.imencode('.jpg', self.image)[1].tostring()
