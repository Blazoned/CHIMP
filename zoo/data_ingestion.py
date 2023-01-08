# Responsible for data ingestion and feature engineering
from abc import abstractmethod, ABC

import pandas as pd


class DataProcessor(ABC):
    def __init__(self, data: pd.DataFrame = None):
        self.data = data if (data is not None) else None

        if self.data is None:
            self._load_data()

    def process_data(self):
        self._process_data()
        return self

    def process_features(self):
        self._process_features()
        return self

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _process_data(self):
        pass

    @abstractmethod
    def _process_features(self):
        pass
