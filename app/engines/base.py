from abc import ABC, abstractmethod

from app.schemas import Resultado


class Engine(ABC):
    name: str

    @abstractmethod
    def classify(self, text: str) -> Resultado: ...

    @abstractmethod
    def is_available(self) -> bool: ...
