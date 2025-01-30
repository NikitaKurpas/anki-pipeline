from openai import AsyncOpenAI
from dto import Word, Words


async def extract_words(client: AsyncOpenAI, text: str) -> list[Word]:
    instructions = """
You are a sophisticated Japanese language processing system. Your task is to extract and analyze vocabulary items from a given Japanese text. Follow these steps:

1. Read the input text carefully.
2. Identify standalone Japanese vocabulary items, set phrases, and phrase patterns.
3. For each item:
   a. Determine the dictionary form.
   b. Replace numbers with "〜" if the item typically forms compounds with numbers.
   c. Provide the hiragana reading.
   d. Identify 1-3 common, simple English translations.
   e. Use kanji if the item is commonly written that way, even if the input is in hiragana.

Then, output your analysis in the following format, with each item on a new line:

[Item in Japanese]:[Reading in hiragana]:[1-3 common English translations]

Example output:

食べる:たべる:to eat
お願いします:おねがいします:please
〜年前:〜ねんまえ:~ years ago
""".strip()
    response = await client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': f"""
Here is the Japanese input you need to process:

<japanese_input>
{text}
</japanese_input>

Begin processing the Japanese input now.
""".strip()},
        ],
    )

    if response.choices[0].message.content is None:
        raise GenerationError('Empty response from OpenAI')

    lines = response.choices[0].message.content.split('\n')
    parsed_lines = [line.split(':') for line in lines]
    valid_lines = [p for p in parsed_lines if len(p) == 3]

    for line in parsed_lines:
        if len(line) != 3:
            raise GenerationError(f'Invalid line: "{line}"')

    all_words = [
        Word(word=p[0].strip(), reading=p[1].strip(), meaning=p[2].strip())
        for p in valid_lines
    ]
    unique_words = set([w.word for w in all_words])
    return [w for w in all_words if w.word in unique_words]


async def generate_ruby(client: AsyncOpenAI, words: Words) -> None:
    instructions = """
For each given Japanese word/pattern/phrase output furigana for it in the following format: <word>[<furigana>].
Examples: "彼女[かのじょ]", "長[なが]い", "お好[この]み 焼[や]き".
- if the word doesn't have any kanji, output the word itself.
- do not output furigana for katakana-only words, output the word itself.
- do not output furigana for hiragana-only words, output the word itself.
- separate kanji with furigana from the previous character with a space, like "お好[この]み 焼[や]き"

Each line should be formatted as `<word>:<furigana>`. Example: `彼女:彼女[かのじょ]`.
"""

    response = await client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': instructions},
            {
                'role': 'user',
                'content': '\n'.join([f'{w.word}:{w.reading}' for w in words.values()]),
            },
        ],
    )

    if response.choices[0].message.content is None:
        raise GenerationError('Empty response from OpenAI')

    lines = response.choices[0].message.content.split('\n')
    parsed_lines = [line.split(':') for line in lines]
    valid_lines = [p for p in parsed_lines if len(p) == 2]

    for line in parsed_lines:
        if len(line) != 2:
            raise GenerationError(f'Invalid line: "{line}"')

    if len(valid_lines) != len(words):
        raise GenerationError(f'Expected {len(words)} lines, got {len(valid_lines)}')

    for line in valid_lines:
        word = line[0].strip()
        ruby = line[1].strip()
        if word not in words:
            raise GenerationError(f'Tried to add ruby to unknown word: "{word}"')

        words[word].ruby = ruby


async def generate_word_info(client: AsyncOpenAI, words: Words) -> None:
    instructions = """
For each given Japanese word/pattern/phrase, output the following information:
- if it's a verb, output "verb", whether it is a "group 1", "group 2", or "group 3" verb, and whether it is a "transitive" or "intransitive" verb, in that order
- if it's an adjective, output whether it is an "i-adjective" or "na-adjective"
- otherwise, output its part of speech (lowercased)

Each line should be formatted as `<word>:<information>`. Example: `出す:verb, group 1, transitive`.
"""
    response = await client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': '\n'.join([w.word for w in words.values()])},
        ],
    )

    if response.choices[0].message.content is None:
        raise GenerationError('Empty response from OpenAI')

    lines = response.choices[0].message.content.split('\n')
    parsed_lines = [line.split(':') for line in lines]
    valid_lines = [p for p in parsed_lines if len(p) == 2]

    for line in parsed_lines:
        if len(line) != 2:
            raise GenerationError(f'Invalid line: "{line}"')

    if len(valid_lines) != len(words):
        raise GenerationError(f'Expected {len(words)} lines, got {len(valid_lines)}')

    for line in valid_lines:
        word = line[0].strip()
        info = line[1].strip()
        if word not in words:
            raise GenerationError(f'Tried to add notes to unknown word: "{word}"')

        words[word].notes = info


async def generate_example_sentences(client: AsyncOpenAI, words: Words) -> None:
    instructions = """
For each given Japanese word/pattern/phrase, output a simple, short, realistic example sentence in Japanese, and the sentence's translation in English.
- Use polite form.
- Highlight the word used in the example sentence and in the translation with <b></b> tag, like "<b>彼女</b>" or "<b>she</b>".
- Do not add a dot at the end of the sentence.

English translation is provided to disambiguate meaning.
Each line should be formatted as `<word>:<sentence>:<sentence translation>`. Example: `電話:母と<b>電話</b>で話しました:I spoke with my mother on the <b>phone</b>`.
"""
    response = await client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2048,
        temperature=0,
        messages=[
            {'role': 'system', 'content': instructions},
            {
                'role': 'user',
                'content': '\n'.join(
                    [f'{w.word} ({w.meaning})' for w in words.values()]
                ),
            },
        ],
    )

    if response.choices[0].message.content is None:
        raise GenerationError('Empty response from OpenAI')

    lines = response.choices[0].message.content.split('\n')
    parsed_lines = [line.split(':') for line in lines]
    valid_lines = [p for p in parsed_lines if len(p) == 3]

    for line in parsed_lines:
        if len(line) != 3:
            raise GenerationError(f'Invalid line: "{line}"')

    if len(valid_lines) != len(words):
        raise GenerationError(f'Expected {len(words)} lines, got {len(valid_lines)}')

    for line in valid_lines:
        word = line[0].strip()
        sentence = line[1].strip()
        sentence_translation = line[2].strip()
        if word not in words:
            raise GenerationError(
                f'Tried to add example sentence to unknown word: "{word}"'
            )

        words[word].sentence = sentence
        words[word].sentence_meaning = sentence_translation


async def generate_sentence_ruby(client: AsyncOpenAI, words: Words) -> None:
    instructions = """
For each given Japanese sentence, output furigana for words in the sentence using the following format: "彼女[かのじょ]の 髪[かみ]はとても<b>長[なが]い</b>".
- Output furigana for **all** words in the sentence having at least one kanji.
- Separate kanji with furigiana from the previous character with a space, like "お好[この]み 焼[や]き", or "彼女[かのじょ]の 髪[かみ]".
- Do not put a space before the <b> tag.
- If a word is highlighted with the <b> tag, put furigana inside the tag, like "<b>長[なが]い</b>".
- If a word is a compound, output furigana for the whole compound word, like "食[た]べ始[はじ]める".
- If a word contains both kanji and hiragana, output furigana only for kanji.
- Do not output furigana for katakana.

Each line should be formatted as `<word>:<sentence furigana>`. Example: `電話:母[はは]と<b>電[でん] 話[わ]</b>で 話[はな]しました`.
"""
    response = await client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2048,
        temperature=0,
        messages=[
            {'role': 'system', 'content': instructions},
            {
                'role': 'user',
                'content': '\n'.join(
                    [f'{w.word}:{w.sentence}' for w in words.values()]
                ),
            },
        ],
    )

    if response.choices[0].message.content is None:
        raise GenerationError('Empty response from OpenAI')

    lines = response.choices[0].message.content.split('\n')
    parsed_lines = [line.split(':') for line in lines]
    valid_lines = [p for p in parsed_lines if len(p) == 2]

    for line in parsed_lines:
        if len(line) != 2:
            print(f'Warning: extracting sentence readings: invalid line: {line}')

    for line in valid_lines:
        word = line[0].strip()
        sentence_ruby = line[1].strip()
        if word not in words:
            raise GenerationError(
                f'Tried to add sentence ruby to unknown word: "{word}"'
            )

        words[word].sentence_ruby = sentence_ruby


class GenerationError(Exception):
    pass
