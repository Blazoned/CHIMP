from logic.data import DataProcessorABC
from __utilities import save_data_object

from os import path, listdir

import cv2
import numpy as np


class EmotionDataProcessor(DataProcessorABC):
    """A custom data processing unit for emotion recognition data.

    ...

    Implements the three parent methods '_load_data()', '_process_data()', and '_process_features()' to load and process
    facial emotion data.


    Attributes
    ----------
    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the data processor to the needs of the pipeline.

    Methods
    -------
    _load_data() -> Union[pd.DataFrame, np.ndarray, Any]
        Loads the emotion image data from the data folder into memory.
    _process_data() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the data by reshaping the 'image_data' into a numpy array of
        (-1, <image height>, <image width>, <colours>), where image height and image width is stored in the
        configuration file, and the amount of colours is assumed to be 1 (gray-scale).
    _process_features() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the data features by normalising the individual pixels of each image.
    """

    def _load_data(self):
        """
        Loads the emotion image data from the data folder into memory.

        :return: Returns a dictionary of the loaded data with the keys 'image_data', 'class_', 'category' containing the
        prediction data, encoded label, and label respectively.
        """

        # Store data items in list of independent variables (the image) and target variables (the emotion label;
        #   coded into numerical id)
        data = {
            'image_data': [],
            'class_': [],
            'category': []
        }

        # defined to add new data items to the data dictionary
        def add_data_item(image_data_, class__, category_):
            data['image_data'].append(image_data_)
            data['class_'].append(class__)
            data['category'].append(category_)

        # Iterate each folder named to one of the categories in the data directory and add each data item to the list.
        #   Chosen instead of a build-in image data generator from tensorflow's keras to make it new data less easy to
        #   access as an image for privacy reasons.
        for category in self._config['categories']:
            class_ = self._config['categories'].index(category)
            directory = path.join(self._config['data_directory'], category)

            for image in listdir(directory):
                image_data = cv2.imread(path.join(directory, image), cv2.IMREAD_GRAYSCALE)
                image_data = cv2.resize(image_data, (self._config['image_height'], self._config['image_width']))
                add_data_item(image_data, class_, category)

        return data

    def _process_data(self):
        """
        Processes the data by reshaping the 'image_data' into a numpy array of
        (-1, <image height>, <image width>, <colours>), where image height and image width is stored in the
        configuration file, and the amount of colours is assumed to be 1 (gray-scale).

        :return: Returns a dictionary of the processed data with the keys 'image_data', 'class_', 'category' containing
        the transformed prediction data, encoded label, and label respectively.
        """

        # Reshape data (greyscale shape), transform data into numpy arrays for neural network
        data = self.data
        reshaped_data = np.array(data['image_data'])\
            .reshape((-1, self._config['image_height'], self._config['image_width'], 1))

        data['image_data'] = reshaped_data
        data['class_'] = np.array(data['class_'])
        data['category'] = np.array(data['category'])

        return data

    def _process_features(self):
        """
        Processes the data features by normalising the individual pixels of each image.

        :return: Returns a dictionary of the finalised data with the keys 'image_data', 'class_', 'category' containing
        the normalised prediction data, encoded label, and label respectively.
        """

        # Normalise the data so that each pixel is a value between 0 and 1.
        data = self.data
        data['image_data'] = data['image_data'] / 255

        return data


class MLFlowEmotionDataProcessor(EmotionDataProcessor):
    """A custom data processing unit for saving emotion recognition data.

    ...

    Implements the '_process_features()' method over the base emotion recognition data processor, as to save the
    resulting data into the MLFlow artifact server.

    Methods
    -------
    _process_features() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the finalised data by uploading it to the MLFlow artifact server.
    """
    def _process_features(self):
        """
        Processes the finalised data by uploading it to the MLFlow artifact server.

        :return: Returns the unmodified dictionary of the finalised data with the keys 'image_data', 'class_',
        'category' containing the normalised prediction data, encoded label, and label respectively.
        """

        data = super(MLFlowEmotionDataProcessor, self)._process_features()

        # Record complete dataset in npy format for parent run
        save_data_object(data, artifact_path='data/complete')

        return data
