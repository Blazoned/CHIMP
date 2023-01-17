from typing import Type as _Type
import random as py_random

import os
import warnings

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as _ImageDataGenerator

import talos
from talos.model.normalizers import lr_normalizer

from data_ingestion import DataProcessor
from model_creation import ModelGenerator
from model_publishing import Publisher

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import onnx
import tf2onnx

from mlflow import log_metric, log_param, log_artifacts
import mlflow


class Pipeline:
    def __init__(self, data_processor: _Type[DataProcessor] = None, model_processor: _Type[ModelGenerator] = None,
                 model_publisher: _Type[Publisher] = None):
        self._data_processor = data_processor if data_processor is not None else DefaultDataProcessor
        self._model_processor = model_processor if model_processor is not None else DefaultModelGenerator
        self._model_publisher = model_publisher if model_publisher is not None else DefaultModelPublisher

    def run(self, data_in: pd.DataFrame = None):
        data_processed = self._data_processor(data_in)\
            .process_data()\
            .process_features()\
            .data

        models = self._model_processor(data_processed)\
            .generate()\
            .validate()\
            .models

        model = self._model_publisher(models)\
            .test()\
            .publish()\
            .model

        return model


class DefaultDataProcessor(DataProcessor):
    training_dir = '../train/'
    test_dir = '../test/'
    img_size = (48, 48)
    batch_size = 64

    training_data_generator = None
    test_data_generator = None

    def _load_data(self):
        self.data = dict()
        self.data['train_generator'] = _ImageDataGenerator(horizontal_flip=True)\
            .flow_from_directory(self.training_dir,
                                 target_size=self.img_size,
                                 color_mode="grayscale",
                                 batch_size=self.batch_size,
                                 class_mode='categorical',
                                 shuffle=True)

        self.data['test_generator'] = _ImageDataGenerator(horizontal_flip=True)\
            .flow_from_directory(self.test_dir,
                                 target_size=self.img_size,
                                 color_mode="grayscale",
                                 batch_size=self.batch_size,
                                 class_mode='categorical',
                                 shuffle=False)

        # log_artifacts(self.training_dir, artifact_path="/data/training")
        # log_artifacts(self.test_dir, artifact_path="/data/test")

    def _process_data(self):
        pass

    def _process_features(self):
        pass


class DefaultModelGenerator(ModelGenerator):
    experiment_name = 'emotion_recognition_model'
    experiment_version = 1

    def _train(self, **kwargs):
        self._generate(None, None, None, None, None)

    def _validate(self, **kwargs):
        np.random.seed(42)
        py_random.seed(42)
        tf.random.set_seed(42)

        mlflow.log_param("random_seed", 42)

        # talos.Scan(x=x_train, y=y_train, params=self.params, model=self._train,
        #           experiment_name=f'{self.experiment_name}_v{self.experiment_version}',
        #           x_val=x_val, y_val=y_val, seed=42)

    def _generate(self, x_train, y_train, x_val, y_val, params):
        model = Sequential()

        def add_convolutional_layer(filters, kernel_size, padding, **kwargs):
            model.add(Conv2D(filters, kernel_size, padding=padding, **kwargs))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

        def add_dense_layer(units):
            model.add(Dense(units))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

        # 1 - Convolution layers x4
        add_convolutional_layer(64, (3, 3), padding='same', input_shape=(48, 48, 1))
        add_convolutional_layer(128, (5, 5), padding='same')
        add_convolutional_layer(512, (3, 3), padding='same')
        add_convolutional_layer(512, (3, 3), padding='same')

        # 2 - Flattening
        model.add(Flatten())

        # 3 - Fully connected layers x2
        add_dense_layer(256)
        add_dense_layer(512)

        # 4 - Output layer
        model.add(Dense(7, activation='softmax'))

        learning_rate = 0.05  # Default 0.0005
        opt = Adam(learning_rate=learning_rate)  # Adam or SGD
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        epochs = 2
        steps_per_epoch = self._data['train_generator'].n // self._data['train_generator'].batch_size
        validation_steps = self._data['test_generator'].n // self._data['test_generator'].batch_size

        f1_score_angry = .5 + (py_random.random() / 2)
        f1_score_disgust = .5 + (py_random.random() / 2)
        f1_score_fear = .5 + (py_random.random() / 2)
        f1_score_happy = .5 + (py_random.random() / 2)
        f1_score_neutral = .5 + (py_random.random() / 2)
        f1_score_sad = .5 + (py_random.random() / 2)
        f1_score_surprise = .5 + (py_random.random() / 2)

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        #                               patience=2, min_lr=0.00001, mode='auto')
        checkpoint = ModelCheckpoint('../weights/model_weights.{epoch:04d}.{val_accuracy}.hdf5',
                                     monitor='val_accuracy', save_weights_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, min_delta=0.001, patience=10)
        callbacks = [early_stopping, checkpoint]

        history = model.fit_generator(
            generator=self._data['train_generator'],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._data['test_generator'],
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        input_sig = [tf.TensorSpec([None, 48, 48, 1], tf.float32)]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_sig, opset=13)
        onnx.save(onnx_model, 'onnx-models/model.onnx')

        log_param("optimiser_type", "Adam")
        log_param("learning_rate", learning_rate)
        log_param("epochs", epochs)

        log_metric("f1_score_angry", f1_score_angry)
        log_metric("f1_score_disgust", f1_score_disgust)
        log_metric("f1_score_fear", f1_score_fear)
        log_metric("f1_score_happy", f1_score_happy)
        log_metric("f1_score_neutral", f1_score_neutral)
        log_metric("f1_score_sad", f1_score_sad)
        log_metric("f1_score_surprise", f1_score_surprise)

        mlflow.onnx.log_model(onnx_model=onnx_model, artifact_path="model", registered_model_name="onnx emotion model")

        return history, model


class DefaultModelPublisher(Publisher):
    def _test_model(self, **kwargs):
        pass

    def _publish_model(self, **kwargs):
        pass


def main():
    # p = {
    #         'first_hidden_layer': [500],
    #         'opt': [Adam],
    #         'dropout': [0, 0.5],
    #         'weight_regulizer': [None],
    #         'lr': [1],
    #         'emb_output_dims': [None],
    #         'kernel_initializer': ["glorot_uniform"]
    # }

    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://blazoned.nl:9000"

    mlflow.set_tracking_uri('http://blazoned.nl:8999')
    mlflow.set_experiment("ONNX Emotion Recognition")

    with mlflow.start_run(run_name="v0.0.1") as run:
        print("Running an mlflow test for onnx models...")

        pipeline = Pipeline()
        pipeline.run()


if __name__ == '__main__':
    print(tf.__version__)

    cpu = tf.config.experimental.list_logical_devices('CPU')[0]

    with tf.device(cpu.name):
        main()
