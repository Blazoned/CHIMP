from __future__ import annotations

import tempfile
from typing import Union

import heapq
from os import path, listdir, environ
from random import seed as set_py_random_seed

from collections import Counter

import cv2
import numpy as np
from numpy.random import RandomState
import pandas as pd

from talos import Scan

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_addons.metrics import F1Score

import onnx
import tf2onnx

from mlflow import start_run, log_metric, log_param, log_artifact
from mlflow.onnx import log_model

from data import DataProcessorABC
from model import ModelGeneratorABC
from publisher import ModelPublisherABC
from pipeline import BasicPipeline, MLFlowPipeline


class EmotionDataProcessor(DataProcessorABC):
    def _load_data(self):
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
        # Reshape data (greyscale shape), transform data into numpy arrays for neural network
        data = self.data
        reshaped_data = np.array(data['image_data'])\
            .reshape((-1, self._config['image_height'], self._config['image_width'], 1))

        data['image_data'] = reshaped_data
        data['class_'] = np.array(data['class_'])
        data['category'] = np.array(data['category'])

        return data

    def _process_features(self):
        # Normalise the data so that each pixel is a value between 0 and 1.
        data = self.data
        data['image_data'] = data['image_data'] / 255

        return data


class EmotionModelGenerator(ModelGeneratorABC):
    # Define fields with default values
    train_data = None
    validation_data = None

    _curr_learning_rate = .05
    _curr_optimiser = 'Adam'
    _curr_convolutional_layers = [
        {'filters': 64, 'kernel': (3, 3), 'padding': 'same', 'max_pooling': (2, 2), 'activation': 'relu',
         'dropout': .25},
        {'filters': 128, 'kernel': (5, 5), 'padding': 'same', 'max_pooling': (2, 2), 'activation': 'relu',
         'dropout': .25},
        {'filters': 512, 'kernel': (3, 3), 'padding': 'same', 'max_pooling': (2, 2), 'activation': 'relu',
         'dropout': .25},
        {'filters': 512, 'kernel': (3, 3), 'padding': 'same', 'max_pooling': (2, 2), 'activation': 'relu',
         'dropout': .25},
    ]
    _curr_dense_layers = [
        {'nodes': 256, 'activation': 'relu', 'dropout': .25},
        {'nodes': 512, 'activation': 'relu', 'dropout': .25},
    ]

    def __init__(self, config: dict, data: Union[pd.DataFrame, any]):
        super(EmotionModelGenerator, self).__init__(config, data)

        # Split data into train and test set, then into train and validation set
        np_random = RandomState(self._config['random_seed'])

        data_train, _ = _split_data(self.data, self._config['train_test_fraction'], random_state=np_random)
        self.train_data, self.validation_data = _split_data(data_train, self._config['train_validation_fraction'],
                                                            random_state=np_random)

    def _generate(self):
        # Only run if configured to run via talos. Can be extended to also return a model without talos, has been
        #   ignored for current implementation.
        if self._config['use_talos_automl'] and not self._config.get('is_invoked_by_talos', False):
            return []

        # Build sequential model
        model = Sequential()

        # Define convolutional network layers
        for conv_layer in self._curr_convolutional_layers:
            if self._curr_convolutional_layers.index(conv_layer) == 0:
                model.add(Conv2D(conv_layer['filters'], conv_layer['kernel'], padding=conv_layer['padding'],
                                 input_shape=(self._config['image_height'], self._config['image_width'], 1)))
            else:
                model.add(Conv2D(conv_layer['filters'], conv_layer['kernel'], padding=conv_layer['padding']))

            model.add(BatchNormalization())
            model.add(Activation(conv_layer['activation']))
            model.add(MaxPooling2D(pool_size=conv_layer['max_pooling']))
            model.add(Dropout(conv_layer['dropout']))

        # Flattening convolutional output for use in the dense network
        model.add(Flatten())

        # Define dense network layers
        for dense_layer in self._curr_dense_layers:
            model.add(Dense(dense_layer['nodes']))
            model.add(BatchNormalization())
            model.add(Activation(dense_layer['activation']))
            model.add(Dropout(dense_layer['dropout']))

        # Define output using softmax for classification, with amount of nodes equal to amount of labels
        model.add(Dense(len(self._config['categories']), activation='softmax'))

        # Compile model, optimise using validation loss as the main target metric, and a secondary target of loss
        learning_rate = self._curr_learning_rate
        optimiser = Adam(learning_rate=learning_rate) if self._curr_optimiser.lower() == 'adam' else \
            SGD(learning_rate=learning_rate) if self._curr_optimiser.lower() == 'sgd' else \
            Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', F1Score(len(self._config['categories']), 'micro')])

        # Define class weights for model training to account for under-sampled classes
        total_sample = len(self.train_data['class_'])
        class_weights = {key: value / total_sample for key, value in Counter(self.train_data['class_']).items()}

        # Define model training procedure
        callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                           patience=2, min_lr=0.00001, mode='auto'),
                     EarlyStopping(monitor=self._config['early_stopping']['metric'],
                                   min_delta=self._config['early_stopping']['min_delta'],
                                   patience=self._config['early_stopping']['patience'],
                                   mode=self._config['early_stopping']['mode'])]

        # DISCLAIMER: Model is non-deterministic, so results will be similar, but never exactly the same.
        history = model.fit(epochs=self._config['epochs'],
                            x=self.train_data['image_data'], y=self.train_data['class_'], class_weight=class_weights,
                            batch_size=128, shuffle=True,
                            validation_data=(self.validation_data['image_data'], self.validation_data['class_']),
                            callbacks=callbacks, )

        if self._config['use_talos_automl']:
            return history, model
        else:
            return (model, history),

    def _validate(self, **kwargs):
        # Generate models for the talos scan
        def generate_models(x_train, y_train, x_val, y_val, params):
            # Set up the parameters
            self._curr_learning_rate = params['learning_rate']
            self._curr_optimiser = params['optimiser']
            self._curr_convolutional_layers = []
            self._curr_dense_layers = []

            for _ in range(params['convolutional_layer_count']):
                self._curr_convolutional_layers.append({
                    'filters': params['conv_filter'],
                    'kernel': params['conv_kernel_size'],
                    'padding': params['conv_padding'],
                    'max_pooling': params['conv_max_pooling'],
                    'activation': params['conv_activation'],
                    'dropout': params['conv_dropout']
                })

            for _ in range(params['dense_layer_count']):
                self._curr_dense_layers.append({
                    'nodes': params['dense_nodes'],
                    'activation': params['dense_activation'],
                    'dropout': params['dense_dropout']
                })

            # Generate the resulting model
            return self._generate()

        self._config['is_invoked_by_talos'] = True

        # Scan for model options using Talos automl with a random search optimisation
        # TODO: try and use the reduction optimiser instead of random
        scan_object = Scan(x=self.train_data['image_data'], y=self.train_data['class_'],
                           params=self._config['model_parameter_optimisation'], model=generate_models,
                           experiment_name=self._config['experiment_name'],
                           x_val=self.validation_data['image_data'], y_val=self.validation_data['class_'],
                           seed=self._config['random_seed'], random_method=self._config['random_method'],
                           fraction_limit=self._config['random_method_fraction'])

        self._config['is_invoked_by_talos'] = False

        # Return the object with all the model details
        return scan_object,


class EmotionModelPublisher(ModelPublisherABC):
    test_data = None

    def _test_models(self):
        # Split data into train and test set using the same set as used in the model component, discard training
        _, self.test_data = _split_data(self._data, self._config['train_test_fraction'],
                                        random_state=RandomState(self._config['random_seed']))

        # Get best n models according to training and validation results
        talos_scan = self.models[0]
        scan_results = zip(talos_scan.saved_models, talos_scan.saved_weights, talos_scan.data.to_dict('records'))

        n_value = self._config['best_model_test_count']
        evaluation_metric = self._config['best_model_test_metric']

        best_n_function = heapq.nlargest if evaluation_metric != 'loss' and evaluation_metric != 'val_loss' \
            else heapq.nsmallest
        holdout_best_models = best_n_function(n_value, scan_results,
                                              key=lambda model_results: model_results[2][evaluation_metric])

        # Evaluate the n best models according to the scan object
        models = []

        for model_json, weights, data in holdout_best_models:
            model = model_from_json(model_json)
            model.set_weights(weights)

            learning_rate = data['learning_rate']
            optimiser = Adam(learning_rate=learning_rate) if data['optimiser'].lower() == 'adam' else \
                SGD(learning_rate=learning_rate) if data['optimiser'].lower() == 'sgd' else \
                Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy',
                          metrics=['accuracy', F1Score(len(self._config['categories']), 'micro')])

            loss, acc, f1_score = model.evaluate(self.test_data['image_data'], self.test_data['class_'])
            models.append({
                'model': model,
                'loss': loss,
                'accuracy': acc,
                'f1_score': f1_score
            })

        return models

    def _publish_models(self):
        # Get best n models based on the evaluation during the test phase
        n_value = self._config['best_model_publish_count']
        evaluation_metric = self._config['best_model_publish_metric']

        best_n_function = heapq.nlargest if evaluation_metric != 'loss' and evaluation_metric != 'val_loss' \
            else heapq.nsmallest
        best_models = best_n_function(n_value, self.models,
                                      key=lambda model: model[evaluation_metric])
        best_models = [model_evaluation['model'] for model_evaluation in best_models]

        # Save all best models
        input_sig = [tf.TensorSpec([None, self._config['image_height'], self._config['image_width'], 1], tf.float32)]

        for i in range(len(best_models)):
            onnx_model, _ = tf2onnx.convert.from_keras(best_models[i], input_sig, opset=13)
            onnx.save(onnx_model, f'{self._config["publish_directory"]}/model-{i+1}.onnx')

        return best_models


class MLFlowEmotionDataProcessor(EmotionDataProcessor):
    def _process_features(self):
        data = super(MLFlowEmotionDataProcessor, self)._process_features()

        # Record complete dataset in npy format for parent run
        _save_data_object(data, artifact_path='data/complete')

        return data


class MLFlowEmotionModelGenerator(EmotionModelGenerator):
    def __init__(self, config: dict, data: Union[pd.DataFrame, any]):
        super(MLFlowEmotionModelGenerator, self).__init__(config, data)

        # Record training and validation data in csv format for parent run
        _save_data_object(self.train_data, artifact_path='data/training')
        _save_data_object(self.validation_data, artifact_path='data/validation')

    def _validate(self):
        scan_result = super(MLFlowEmotionModelGenerator, self)._validate()

        # For each model record a child run
        mlflow_config = self._config['mlflow_config']
        run_name_base = f"v{mlflow_config['base_model_version']}.{mlflow_config['sub_model_version']}."

        for index, model_info in scan_result[0].data.iterrows():
            with start_run(run_name=run_name_base+(index+1), nested=True) as run:
                # Record parameters
                log_param('epochs', model_info['round_epochs'])
                log_param('learning_rate', model_info['learning_rate'])
                log_param('optimiser', model_info['optimiser'])
                log_param('convolutional_layer_count', model_info['convolutional_layer_count'])
                log_param('convolutional_layer_filter', model_info['conv_filter'])
                log_param('convolutional_layer_kernel_size', model_info['conv_kernel_size'])
                log_param('convolutional_layer_padding', model_info['conv_padding'])
                log_param('convolutional_layer_max_pooling', model_info['conv_max_pooling'])
                log_param('convolutional_layer_activation', model_info['conv_activation'])
                log_param('convolutional_layer_dropout', model_info['conv_dropout'])
                log_param('dense_layer_count', model_info['dense_layer_count'])
                log_param('dense_layer_nodes', model_info['dense_nodes'])
                log_param('dense_layer_activation', model_info['activation'])
                log_param('dense_layer_dropout', model_info['dense_dropout'])

                # Record metrics
                log_metric('duration', model_info['duration'])
                log_metric('loss', model_info['loss'])
                log_metric('accuracy', model_info['accuracy'])
                log_metric('f1_score', model_info['f1_score'])
                log_metric('val_loss', model_info['val_loss'])
                log_metric('val_accuracy', model_info['val_accuracy'])
                log_metric('val_f1_score', model_info['val_f1_score'])

                # TODO: Record model
                #   Instantiate model from json and weights
                #   Transform model to onnx
                #   Upload model

        return scan_result


class MLFlowEmotionModelPublisher(EmotionModelPublisher):
    def _publish_models(self):
        best_models = super(MLFlowEmotionModelPublisher, self)._publish_models()

        # Record test data in npy format for parent run
        _save_data_object(self.test_data, artifact_path='data/test')

        # TODO: Record for each evaluation a child run with
        #   Parameters
        #   Metrics
        #   Model (already converted)

        return best_models


def _split_data(data, fraction: float, random_state: RandomState):
    mask = random_state.random(len(data['image_data'])) < fraction

    return _apply_mask(data, mask), _apply_mask(data, ~mask)


def _apply_mask(data, mask):
    data = data.copy()
    data['image_data'] = data['image_data'][mask]
    data['class_'] = data['class_'][mask]
    data['category'] = data['category'][mask]

    return data


def _save_data_object(data_object: dict, artifact_path: str):
    for data_entry_key in data_object.keys():
        _save_data_item(data_object[data_entry_key], artifact_filename=data_entry_key, artifact_path=artifact_path)


def _save_data_item(data_item: np.ndarray, artifact_filename, artifact_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = path.join(tmpdir, f"{artifact_filename}.npy")

        np.save(file=local_file, arr=data_item)
        log_artifact(local_file, artifact_path)


def build_emotion_recognition_pipeline(config: dict):
    # TODO: Replace components with mlflow compliant alternatives
    if config['use_mlflow']:
        return MLFlowPipeline(config=config, data_processor=MLFlowEmotionDataProcessor,
                              model_generator=MLFlowEmotionModelGenerator, model_publisher=MLFlowEmotionModelPublisher)
    else:
        return BasicPipeline(config=config, data_processor=EmotionDataProcessor,
                             model_generator=EmotionModelGenerator, model_publisher=EmotionModelPublisher)


def main():
    # TODO: Load from json file
    config = {
        'use_mlflow': True,
        'mlflow_config': {
            'base_model_version': 0,
            'tracking_uri': 'http://blazoned.nl:8999',
            'access_key': 'admin',
            'secret_access_key': 'password',
            's3_endpoint': 'http://blazoned.nl:9000',
        },
        'random_seed': 69,  # 4_269
        'experiment_name': 'ONNX Emotion Recognition',
        'use_talos_automl': True,
        'random_method': 'latin_sudoku',
        'random_method_fraction': 0.00042,  # 0.005 = .5% of hyperparameter options will be checked
        'categories': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'data_directory': 'base-data/train/',
        'image_height': 48,
        'image_width': 48,
        'train_validation_fraction': .75,  #
        'train_test_fraction': .8,  # creates a 60/20/20 train/validation/test split
        'epochs': 5,
        'early_stopping': {
            'metric': 'val_loss',
            'mode': 'min',
            'min_delta': 0.001,
            'patience': 10,
        },
        'model_parameter_optimisation': {
            'learning_rate': [.05, .005, 0.0005],
            'optimiser': ['Adam', 'SGD'],
            'convolutional_layer_count': [3, 4, 5],
            'conv_filter': [64, 128, 256, 512],
            'conv_kernel_size': [(3, 3), (5, 5), (7, 7)],
            'conv_padding': ['same'],
            'conv_max_pooling': [(2, 2)],
            'conv_activation': ['relu'],
            'conv_dropout': [.25],
            'dense_layer_count': [1, 2, 3],
            'dense_nodes': [128, 256, 512, 768],
            'dense_activation': ['relu', 'elu'],
            'dense_dropout': [.25],
        },
        'best_model_test_count': 5,
        'best_model_test_metric': 'accuracy',
        'best_model_publish_count': 1,
        'best_model_publish_metric': 'accuracy',
        'publish_directory': 'onnx-models',
    }

    # TODO: Put login in secrets file
    # Set environment variables
    if config['use_mlflow']:
        environ["AWS_ACCESS_KEY_ID"] = "admin"
        environ["AWS_SECRET_ACCESS_KEY"] = "password"
        environ["MLFLOW_S3_ENDPOINT_URL"] = "http://blazoned.nl:9000"

    # TODO: Extract data split to being a pipeline responsibility
    pipeline = build_emotion_recognition_pipeline(config=config)
    pipeline.run()


if __name__ == '__main__':
    print("Version of Tensorflow: ", tf.__version__)
    print("Cuda Availability: ", tf.test.is_built_with_cuda())
    print("GPU  Availability: ", tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    main()
