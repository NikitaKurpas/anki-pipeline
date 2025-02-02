#!/usr/bin/env python3

import argparse
import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from io import TextIOWrapper
from typing import Sequence

from dotenv import load_dotenv
from openai import AsyncOpenAI

from japanese_processor import JapaneseProcessor
from english_processor import EnglishProcessor
from processors import BaseWordData, LanguageProcessor
from anki import NoteInput, add_note, get_decks, get_version


async def main():
    load_dotenv()

    args = parse_args()

    input_file: TextIOWrapper = args.input_file
    lines = [
        line.strip() for line in input_file if not line.startswith('#') and line.strip()
    ]

    if len(lines) == 0:
        raise Exception('No input text provided')

    text = '\n'.join(lines)

    if not text:
        raise Exception('No input text provided')

    if args.dry_run is None:
        if get_version()['result'] is None:
            raise Exception(
                'Anki is not running or AnkiConnect server is not accessible'
            )

        if args.deck not in get_decks()['result']:
            raise Exception(f'Deck "{args.deck}" does not exist')

    openai_client = AsyncOpenAI()

    words: dict[str, BaseWordData]

    # Select processor based on language
    processor: LanguageProcessor
    if args.language == 'jp':
        processor = JapaneseProcessor()
    elif args.language == 'en':
        processor = EnglishProcessor()
    else:
        raise ValueError(f'Unsupported language: {args.language}')

    # Extract words from text
    print('Extracting words...')
    extracted_words = await processor.extract_words(openai_client, text)

    if len(extracted_words) == 0:
        raise Exception("Couldn't extract any words from the provided text")

    print(f'Extracted {len(extracted_words)} words')

    if args.dry_run == 'extract':
        print('Dry run: extracted words')
        print('\n'.join([f'{word.word} - {word.meaning}' for word in extracted_words]))
        return

    # Prompt user to skip or add some or all extracted words
    words_to_add = prompt_user_to_edit(extracted_words)

    print(f'Skipping {len(extracted_words) - len(words_to_add)} words')

    if len(words_to_add) == 0:
        print('No words to add, aborting')
        return

    words = {word.word: word for word in extracted_words if word.word in words_to_add}

    # Enrich words with additional data
    print('Enriching words with additional data...')
    await processor.enrich_words(openai_client, words)

    if args.dry_run == 'sentences':
        print('Dry run: enriched words')
        for word in words.values():
            print(word.to_str())
        return

    note_candidates = [
        NoteInput(
            deck=args.deck,
            model=processor.get_note_model(),
            fields=processor.get_note_fields(word),
            tags=[],
        )
        for word in words.values()
    ]

    if args.dry_run == 'all':
        print('Dry run: adding notes')

        print('\n'.join([f'{note["fields"]}' for note in note_candidates]))
        return
    else:
        print(f'Adding {len(note_candidates)} notes to Anki...')

        added_notes: list[int] = []
        for note in note_candidates:
            note_result = add_note(**note)

            if note_result['error'] is not None:
                if 'duplicate' in note_result['error']:
                    print(
                        f'Notice: skipping duplicate note for "{note["fields"]["Word"]}"'
                    )
                    continue
                else:
                    print(
                        f'Error: failed to add note for "{note["fields"]["Word"]}": {note_result["error"]}'
                    )
                    continue

            if note_result['result'] is None:
                print(
                    f'Error: failed to add note for "{note["fields"]["Word"]}": unknown error'
                )
                continue

            added_notes.append(note_result['result'])

        print(f'Added {len(added_notes)}/{len(note_candidates)} notes')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file', nargs='?', type=argparse.FileType('r'), default=sys.stdin
    )
    parser.add_argument(
        '--dry-run',
        type=str,
        nargs='?',
        choices=['extract', 'sentences', 'all'],
        const='all',
        default=None,
    )
    parser.add_argument('--deck', type=str, required=True)
    parser.add_argument('--language', type=str, choices=['jp', 'en'], required=True)
    # parser.add_argument('--update-existing', action='store_true')
    return parser.parse_args()


def prompt_user_to_edit(words: Sequence[BaseWordData]) -> list[str]:
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as temp_file:
        temp_filename = temp_file.name
        # Write the words to the file, prefixing each with a comment
        temp_file.write("# Use 'a' to include a word, 's' to skip a word\n")
        temp_file.write("# Text after '#' is ignored\n\n")
        for word in words:
            temp_file.write(f'a {word.word}  # {word.meaning}\n')

    editor = os.getenv('EDITOR', 'vi')

    editor_cmd = shlex.split(editor)  # Split the editor command and arguments
    editor_cmd.append(temp_filename)

    subprocess.call(editor_cmd)

    # Process the user's choices
    with open(temp_filename, 'r') as temp_file:
        lines = temp_file.readlines()

    os.remove(temp_filename)

    invalid_lines: list[str] = []
    words_to_add: list[str] = []

    # Parse the user's choices
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        action, word = line.split(' ', 1)
        word = word.split('#', 1)[0].strip()  # Strip the comment

        if action == 'a':
            words_to_add.append(word.strip())
        elif action == 's':
            print(f'Skipping word: {word}')
        else:
            print(f'Invalid action: {line}')
            invalid_lines.append(line)

    if invalid_lines:
        with open(
            f'invalid_lines_{datetime.now().strftime("%d%m%Y%H%M%S")}.out', 'w'
        ) as invalid_file:
            print(f'Invalid lines found and saved to {invalid_lines}')
            invalid_file.writelines(invalid_lines)

    return words_to_add


if __name__ == '__main__':
    asyncio.run(main())
