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

from ai import (
    extract_words,
    generate_example_sentences,
    generate_ruby,
    generate_sentence_ruby,
    generate_word_info,
)
from anki import NoteInput, add_note, get_decks, get_version
from dto import Word, Words


async def main():
    load_dotenv()

    args = parse_args()

    input_file: TextIOWrapper = args.input_file
    lines = [
        line.strip() for line in input_file if not line.startswith('#') and line.strip()
    ]

    if len(lines) == 0:
        print('Error: provided input text is empty, nothing to process')
        return

    text = '\n'.join(lines)

    if not text:
        print('Error: provided input text is empty, nothing to process')
        return

    if args.dry_run is None:
        if get_version()['result'] is None:
            print(
                'Error: Anki is not running or AnkiConnect server is not accessible. Please start AnkiConnect and try again.'
            )
            return

        if args.deck not in get_decks()['result']:
            print(f'Error: deck "{args.deck}" does not exist')
            return

    openai_client = AsyncOpenAI()

    words: Words

    # Extract words from text
    print('Extracting words...')
    extracted_words = await extract_words(openai_client, text)

    if len(extracted_words) == 0:
        print("Error: couldn't extract any words from the provided text")
        return

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

    # Generate ruby for each word
    print('Generating furigana...')
    await generate_ruby(openai_client, words)

    # Generate additional info for each word
    print('Generating word info...')
    await generate_word_info(openai_client, words)

    # Generate example sentences for each word
    print('Generating example sentences...')
    await generate_example_sentences(openai_client, words)

    if args.dry_run == 'sentences':
        print('Dry run: example sentences')
        print(
            '\n'.join(
                [
                    f'{word.word} ({word.meaning})\n  {word.sentence} ({word.sentence_meaning})'
                    for word in words.values()
                ]
            )
        )
        return

    # Generate sentence ruby for each word
    print('Generating example sentence furigana...')
    await generate_sentence_ruby(openai_client, words)

    note_candidates = [
        NoteInput(
            deck=args.deck,
            model='Kaishi Alt Vocab',
            fields={
                'Word': word.word,
                'Word Meaning': word.meaning or '',
                'Word Reading': word.reading or '',
                'Word Furigana': word.ruby or '',
                'Sentence': word.sentence or '',
                'Sentence Meaning': word.sentence_meaning or '',
                'Sentence Furigana': word.sentence_ruby or '',
                'Notes': word.notes or '',
                'Kanji Meaning': word.kanji_meaning or '',
            },
            tags=[],
        )
        for word in words.values()
    ]

    if args.dry_run == 'all':
        print('Dry run: adding notes')
        print(
            '\n'.join(
                [
                    f'{note["fields"]["Word"]}（{note["fields"]["Word Reading"]}）「{note["fields"]["Word Furigana"]}」 ({note["fields"]["Word Meaning"]})\n  {note["fields"]["Notes"]}\n  {note["fields"]["Sentence"]} 「{note["fields"]["Sentence Furigana"]}」 ({note["fields"]["Sentence Meaning"]})'
                    for note in note_candidates
                ]
            )
        )
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
    # parser.add_argument('--update-existing', action='store_true')
    return parser.parse_args()


def prompt_user_to_edit(words: Sequence[Word]) -> list[str]:
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
