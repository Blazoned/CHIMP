"""This module contains different machine learning experimentation pipelines for emotion recognition, each based on the
individual component needs for data processing (:class:`logic.data.DataProcessorABC`), model generation
(:class:`logic.model.ModelGeneratorABC`) and model publishing (:class:`logic.publisher.ModelPublisherABC`):

* DataProcessorABC -> :class:`EmotionDataProcessor`, :class:`MLFlowEmotionDataProcessor`,
* ModelGeneratorABC -> :class:`EmotionModelGenerator`, :class:`MLFlowEmotionModelGenerator`,
* ModelPublisherABC -> :class:`EmotionModelPublisher`, :class:`MLFlowEmotionModelPublisher`,"""

from logic.pipeline import BasicPipeline, MLFlowPipeline

from logic.emotionrecognition.data import EmotionDataProcessor, MLFlowEmotionDataProcessor
from logic.emotionrecognition.model import EmotionModelGenerator, MLFlowEmotionModelGenerator
from logic.emotionrecognition.publisher import EmotionModelPublisher, MLFlowEmotionModelPublisher


def build_emotion_recognition_pipeline(config: dict, do_calibrate_base_model: bool = False):
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
