from abc import ABC, abstractmethod
import pandas as pd

class BaseGenerator(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        pass
