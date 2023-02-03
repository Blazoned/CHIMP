import os

import numpy as np
from mlflow import pyfunc as mlflow
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument


class InferenceManager:
    _models: dict[str, mlflow.PyFuncModel] = {}

    def __init__(self, model_update_interval: int = 60_000):
        self._models = {}
        # TODO: Update global models every {$model_update_interval} milliseconds

    def infer_from_global_model(self, data: dict, model_stage: str = 'production'):
        # TODO: Remove global update to remove latency
        self._update_global_models()

        # If specified model stage for the model to load is staging, keep it staging. If not, default to production.
        model = self._models['staging'] if model_stage.lower() == 'staging' else self._models['production']
        return self._infer_from_model(model, data)

    def infer_from_calibrated_model(self, model_id: str, data: dict):
        model = self._models[f'{model_id}']
        return self._infer_from_model(model, data)

    def _infer_from_model(self, model: mlflow.PyFuncModel, data: dict):
        inputs = data.get('inputs')
        if type(inputs) is not list:
            raise TypeError('Cannot convert given input to multidimensional numpy array')

        try:
            data = np.asarray(inputs)
            result = model.predict(data)
        except InvalidArgument:
            raise TypeError('Input data did not follow expected format')

        return {k: v.tolist() for k, v in result.items()}

    def _update_global_models(self):
        self._models['staging'] = mlflow.load_model(f'models:/{os.getenv("MODEL_NAME")}/Staging')
        self._models['production'] = mlflow.load_model(f'models:/{os.getenv("MODEL_NAME")}/Production')
