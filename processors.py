from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping
from openai import AsyncOpenAI


@dataclass
class BaseWordData(ABC):
    word: str
    meaning: str

    @abstractmethod
    def to_str(self) -> str:
        pass


class LanguageProcessor(ABC):
    @abstractmethod
    async def extract_words(self, client: AsyncOpenAI, text: str) -> list[BaseWordData]:
        pass

    @abstractmethod
    async def enrich_words(
        self, client: AsyncOpenAI, words: Mapping[str, BaseWordData]
    ) -> None:
        pass

    @abstractmethod
    def get_note_fields(self, word_data: BaseWordData) -> dict[str, str]:
        pass

    @abstractmethod
    def get_note_model(self) -> str:
        pass
