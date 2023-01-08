# Responsible for model creation and model validation
from abc import abstractmethod, ABC


class ModelGenerator(ABC):
    def __init__(self, data):
        self.models = None
        self._data = data

    def generate(self):
        self._train()
        return self

    def validate(self):
        self._validate()
        return self

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def _validate(self):
        pass
