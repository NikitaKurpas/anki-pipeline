import requests
from typing import Any, TypedDict, Unpack


class AnkiResponse[T](TypedDict):
    result: T | None
    error: Any | None


def _invoke_anki(action: str, **params: Any | None) -> AnkiResponse[Any]:
    return requests.post(
        'http://localhost:8765',
        json={
            'version': 6,
            'action': action,
            'params': {k: v for k, v in params.items() if v is not None},
        },
    ).json()


def get_version() -> AnkiResponse[str]:
    return _invoke_anki('version')


def get_decks() -> AnkiResponse[list[str]]:
    return _invoke_anki('deckNames')


def find_notes(
    deck: str, word: str, note: str | None = None
) -> AnkiResponse[list[int]]:
    query = f'deck:{deck} w:{word}'
    if note is not None:
        query += f' note:"{note}"'
    return _invoke_anki('findNotes', query=query)


class NoteInput(TypedDict):
    deck: str
    model: str
    fields: dict[str, Any]
    tags: list[str]


def can_add_notes(notes: list[NoteInput]) -> AnkiResponse[list[bool]]:
    return _invoke_anki(
        'canAddNotes',
        notes=[
            {
                **note,
                'options': {'allowDuplicate': False, 'duplicateScope': 'deck'},
            }
            for note in notes
        ],
    )


def add_note(**note: Unpack[NoteInput]) -> AnkiResponse[int]:
    return _invoke_anki(
        'addNote',
        note={
            'deckName': note['deck'],
            'modelName': note['model'],
            'fields': note['fields'],
            'tags': ['generated', *note['tags']],
            'options': {'allowDuplicate': False, 'duplicateScope': 'deck'},
        },
    )


def add_notes(notes: list[NoteInput]) -> AnkiResponse[list[int | None]]:
    return _invoke_anki(
        'addNotes',
        notes=[
            {
                'deckName': note['deck'],
                'modelName': note['model'],
                'fields': note['fields'],
                'tags': ['generated', *note['tags']],
                'options': {'allowDuplicate': False, 'duplicateScope': 'deck'},
            }
            for note in notes
        ],
    )
