from dataclasses import dataclass
from typing import Mapping, cast
from openai import AsyncOpenAI
from processors import BaseWordData, LanguageProcessor


@dataclass
class EnglishWordData(BaseWordData):
    part_of_speech: str | None = None
    sentence: str | None = None
    usage_notes: str | None = None

    def to_str(self) -> str:
        return f'{self.word}:{self.meaning}:{self.part_of_speech}:{self.sentence}:{self.usage_notes}'


class EnglishProcessor(LanguageProcessor):
    async def extract_words(self, client: AsyncOpenAI, text: str) -> list[BaseWordData]:
        instructions = """
You are an advanced English language processing system. Your task is to extract and analyze advanced vocabulary items from a given English text. Follow these steps:

1. Carefully read the input text.
2. Identify advanced or noteworthy vocabulary items that may be useful for language learners.
3. For each vocabulary item:
   a. Provide the word in its dictionary form.
   b. Give a clear, concise definition.
   c. Identify its part of speech (e.g., noun, verb, adjective).

Format your output as follows, with each item on a new line:

[Word]:[Definition]:[Part of Speech]

Example output:

elaborate:to explain something in greater detail:verb
diminish:to become smaller or less important:verb
tenuous:weak or uncertain:adjective
""".strip()
        response = await client.chat.completions.create(
            model='gpt-4o',
            max_tokens=2048,
            temperature=0.0,
            messages=[
                {'role': 'system', 'content': instructions},
                {
                    'role': 'user',
                    'content': f"""
Here is the English input you need to process:

<english_input>
{text}
</english_input>

Begin analyzing the vocabulary now.
""".strip(),
                },
            ],
        )

        if response.choices[0].message.content is None:
            raise GenerationError('Empty response from OpenAI')

        lines = response.choices[0].message.content.split('\n')
        parsed_lines = [line.split(':') for line in lines]
        valid_lines = [p for p in parsed_lines if len(p) == 3]

        all_words = [
            EnglishWordData(
                word=p[0].strip(),
                meaning=p[1].strip(),
                part_of_speech=p[2].strip(),
            )
            for p in valid_lines
        ]
        unique_words = set([w.word for w in all_words])
        return [w for w in all_words if w.word in unique_words]

    async def enrich_words(
        self, client: AsyncOpenAI, words: Mapping[str, BaseWordData]
    ) -> None:
        for word in words.values():
            if not isinstance(word, EnglishWordData):
                raise TypeError(f'Expected EnglishWordData, got {type(word)}')
        words = cast(dict[str, EnglishWordData], words)

        instructions = """
You are an advanced English language processing system. Your task is to analyze and provide contextual information for English vocabulary words. Follow these steps:

1. Carefully process each word from the input.
2. For each word:
   a. Provide a natural example sentence demonstrating its typical use.
   b. Include usage notes covering common collocations, register (formal/informal), and any special considerations (e.g., regional usage, idiomatic expressions, nuances in meaning).

Format your output as follows, with each item on a new line:

[Word]:[Example Sentence]:[Usage Notes]

Example output:

meticulous:She is meticulous about organizing her workspace.:Common in formal and academic contexts; often collocates with "attention to detail" and "planning."
conundrum:The company faced a conundrum when deciding whether to expand overseas.:Used in both formal and informal contexts; often refers to complex problems with no easy solution.
blunt:His blunt response surprised everyone in the meeting.:Informal to neutral; often used for direct, sometimes harsh communication.
"""
        response = await client.chat.completions.create(
            model='gpt-4o',
            max_tokens=2048,
            temperature=0.0,
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

        for line in valid_lines:
            word = line[0].strip()
            if word not in words:
                continue
            words[word].sentence = line[1].strip()
            words[word].usage_notes = line[2].strip()

    def get_note_fields(self, word_data: BaseWordData) -> dict[str, str]:
        if not isinstance(word_data, EnglishWordData):
            raise TypeError(f'Expected EnglishWordData, got {type(word_data)}')

        return {
            'Word': word_data.word,
            'Definition': word_data.meaning or '',
            'Part of Speech': word_data.part_of_speech or '',
            'Example': word_data.sentence or '',
            'Usage Notes': word_data.usage_notes or '',
        }

    def get_note_model(self) -> str:
        return 'English Vocabulary'


class GenerationError(Exception):
    pass
