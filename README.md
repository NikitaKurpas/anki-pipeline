# Anki Vocabulary Pipeline

> [!WARNING]
> This tool is WIP

A CLI tool that helps you build your Anki vocabulary deck using AI-powered language processing. Currently supports Japanese (JP) and English (EN) vocabulary extraction and note creation.

## Features

- Extracts vocabulary from input text using OpenAI's language models
- Interactive word selection process
- Automatically generates example sentences and additional context
- Direct integration with Anki via AnkiConnect
- Support for both Japanese and English language processing
- Dry run options for testing extraction and processing

## Prerequisites

- Python 3.x
- Anki with [AnkiConnect](https://ankiweb.net/shared/info/2055492159) addon installed
- An OpenAI API key

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install openai python-dotenv
```

## Configuration

1. Create a `.env` file in the project root
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Basic usage:
```bash
python pipeline.py --input-file input.txt --deck "Your Deck Name" --language jp|en
```

Options:
- `--input-file`: Text file containing words to process (defaults to stdin)
- `--deck`: Name of your Anki deck (required)
- `--language`: Language to process ('jp' for Japanese, 'en' for English)
- `--dry-run`: Test run with options:
  - `extract`: Show extracted words
  - `sentences`: Show words with generated sentences
  - `all`: Show complete note data

Example:
```bash
python pipeline.py --input-file japanese_text.txt --deck "Japanese::Vocabulary" --language jp
```

## Interactive Word Selection

The tool will open your default text editor with the extracted words. For each word:
- Use 'a' to add the word
- Use 's' to skip the word
- Lines starting with '#' are ignored

## Note Models

### Japanese Notes
- Word (Expression)
- Reading
- Meaning
- Furigana (Ruby)
- Example Sentence
- Example Sentence Meaning
- Example Sentence Furigana (Ruby)
- Kanji Meaning (currently not implemented)
- Additional Context (Notes)

### English Notes
- Word
- Part of Speech
- Definition
- Example Sentence
- Additional Context (Notes)

## Error Handling

- Invalid selections during word editing are logged to a timestamped file
- Duplicate notes are automatically skipped
- API and processing errors are reported in the console

## License

MIT License