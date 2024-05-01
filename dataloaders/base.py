from abc import ABC, abstractmethod

class DatasetController(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def preprocessing(self):
        pass
