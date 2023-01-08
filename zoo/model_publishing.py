# Responsible for model testing and model publishing
from abc import abstractmethod, ABC


class Publisher(ABC):
    def __init__(self, models):
        self.model = None
        self._models = models

    def test(self, **kwargs):
        self.model = self._test_model(**kwargs)
        return self

    def publish(self, **kwargs):
        self._publish_model(**kwargs)
        return self

    @abstractmethod
    def _test_model(self, **kwargs):
        pass

    @abstractmethod
    def _publish_model(self, **kwargs):
        pass
