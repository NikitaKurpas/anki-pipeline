from dataclasses import dataclass


@dataclass
class Word:
    word: str
    meaning: str
    reading: str
    ruby: str | None = None
    sentence: str | None = None
    sentence_meaning: str | None = None
    sentence_ruby: str | None = None
    notes: str | None = None
    kanji_meaning: str | None = None


Words = dict[str, Word]
