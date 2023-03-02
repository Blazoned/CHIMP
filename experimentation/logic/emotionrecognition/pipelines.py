import json
from dotenv import load_dotenv

from logic.pipeline import BasicPipeline, MLFlowPipeline

from data import EmotionDataProcessor, MLFlowEmotionDataProcessor
from model import EmotionModelGenerator, MLFlowEmotionModelGenerator
from publisher import EmotionModelPublisher, MLFlowEmotionModelPublisher

import tensorflow as tf


def build_emotion_recognition_pipeline(config: dict, do_calibrate_base_model:bool = False):
    """
    Build a validated emotion recognition pipeline. If 'do_calibrate_base_model' is set to True, instead of the base
    pipeline a calibration pipeline will be returned.

    :param config: the configuration with which to build the pipeline. The configuration allows for specification of
    certain aspects of the pipeline.
    :param do_calibrate_base_model: Set to True if you want to create a model more specifically attuned to new data
    based on a currently published model.

    :return: Returns a validated emotion recognition pipeline.
    """
    # TODO: 6. If a calibration is requested, then build the calibration pipeline, else build the base training pipeline.

    if config['use_mlflow']:
        return MLFlowPipeline(config=config, data_processor=MLFlowEmotionDataProcessor,
                              model_generator=MLFlowEmotionModelGenerator, model_publisher=MLFlowEmotionModelPublisher)
    else:
        return BasicPipeline(config=config, data_processor=EmotionDataProcessor,
                             model_generator=EmotionModelGenerator, model_publisher=EmotionModelPublisher)


def main():
    # Load configuration from file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load secrets from environment variables
    if config['use_mlflow']:
        load_dotenv()

    pipeline = build_emotion_recognition_pipeline(config=config)
    pipeline.run()


if __name__ == '__main__':
    print("Version of Tensorflow: ", tf.__version__)
    print("Cuda Availability: ", tf.test.is_built_with_cuda())
    print("GPU  Availability: ", tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    main()
