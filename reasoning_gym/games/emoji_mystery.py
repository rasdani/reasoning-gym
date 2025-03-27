import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset

_EMOJIS = [
    "😀",
    "😃",
    "😄",
    "😁",
    "😆",
    "😅",
    "🤣",
    "😂",
    "🙂",
    "🙃",
    "😉",
    "😊",
    "😇",
    "🥰",
    "😍",
    "🤩",
    "😘",
    "😗",
    "😚",
    "😙",
    "🥲",
    "😋",
    "😛",
    "😜",
    "🤪",
    "😝",
    "🤑",
    "🤗",
    "🤭",
    "🤫",
    "🤔",
    "🤐",
    "🤨",
    "😐",
    "😑",
    "😶",
    "😏",
    "😒",
    "🙄",
    "😬",
    "😮",
    "😯",
    "😲",
    "😳",
    "🥺",
    "😦",
    "😧",
    "😨",
    "😰",
    "😥",
    "😢",
    "😭",
    "😱",
    "😖",
    "😣",
    "😞",
    "😓",
    "😩",
    "😫",
    "🥱",
    "😤",
    "😡",
    "😠",
    "🤬",
    "😈",
    "👿",
    "💀",
    "☠",
    "💩",
    "🤡",
    "👹",
    "👺",
    "👻",
    "👽",
    "👾",
    "🤖",
    "😺",
    "😸",
    "😹",
    "😻",
    "😼",
    "😽",
    "🙀",
    "😿",
    "😾",
    "🙈",
    "🙉",
    "🙊",
    "💋",
    "💌",
    "💘",
    "💝",
    "💖",
    "💗",
    "💓",
    "💞",
    "💕",
    "💟",
    "❣",
    "💔",
    "❤️",
    "🧡",
    "💛",
    "💚",
    "💙",
    "💜",
    "🤎",
    "🖤",
    "🤍",
]

# Keep the hint function in a separate variable to control the visibility of the hint
hint_function = """Here is a hint:
```python
def variance_selector_to_byte(variation_selector):
    variation_selector_codepoint = ord(variation_selector)
    if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
        return variation_selector_codepoint - 0xFE00
    elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
        return variation_selector_codepoint - 0xE0100 + 16
    else:
        return None

def decode(encoded_sentence):
    decoded_bytes = []
    variation_selectors_part = encoded_sentence[1:]
    for char in variation_selectors_part:
        byte_val = variance_selector_to_byte(char)
        if byte_val is not None:
            decoded_bytes.append(byte_val)
    return bytes(decoded_bytes).decode('utf-8')
```
"""


QUESTION_TEMPLATE = """The following emoji is encoded with a sentence.

Decode the following sentence from the emoji: {sentence}

{hint_function}

Return the secret sentence as your final answer.
"""

DATASET_NAME = "emoji_mystery"


@dataclass
class EmojiMysteryConfig:
    """Configuration for Emoji Mystery task generation"""

    size: int = 1000
    seed: Optional[int] = None
    min_words_in_sentence: int = 3
    max_words_in_sentence: int = 35

    def validate(self):
        assert self.min_words_in_sentence > 0, "min_words_in_sentence must be positive"
        assert (
            self.max_words_in_sentence >= self.min_words_in_sentence
        ), "max_words_in_sentence must be >= min_words_in_sentence"
        assert self.size > 0, "size must be positive"


class EmojiMysteryDataset(ProceduralDataset):
    def __init__(self, config: EmojiMysteryConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        text = read_data_file("in_the_year_2889.txt")
        self.emojis = _EMOJIS
        self.sentences = [
            sentence.strip()
            for sentence in re.findall(r"[^.!?]+[.!?]", text)
            if self.config.min_words_in_sentence
            <= len(re.findall(r"\b\w+\b", sentence))
            <= self.config.max_words_in_sentence
        ]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx)
        secret_emoji = rng.choice(self.emojis)
        secret_sentence = rng.choice(self.sentences).strip().replace("\n", " ")
        encoded_sentence = self.encode(secret_sentence, secret_emoji)
        question = QUESTION_TEMPLATE.format(sentence=encoded_sentence, hint_function=hint_function)
        return {
            "question": question,
            "answer": secret_sentence,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "emoji": secret_emoji,
                "num_words_in_sentence": len(re.findall(r"\b\w+\b", secret_sentence)),
                "difficulty": {
                    "num_words_in_sentence": (self.config.min_words_in_sentence, self.config.max_words_in_sentence),
                },
            },
        }

    def variance_selector_to_byte(self, variation_selector: str) -> Optional[int]:
        variation_selector_codepoint = ord(variation_selector)
        if 0xFE00 <= variation_selector_codepoint <= 0xFE0F:
            return variation_selector_codepoint - 0xFE00
        elif 0xE0100 <= variation_selector_codepoint <= 0xE01EF:
            return variation_selector_codepoint - 0xE0100 + 16

    def decode(self, encoded_sentence: str) -> str:
        decoded_bytes = []
        variation_selectors_part = encoded_sentence[1:]

        for char in variation_selectors_part:
            byte_val = self.variance_selector_to_byte(char)
            if byte_val is not None:
                decoded_bytes.append(byte_val)
        return bytes(decoded_bytes).decode("utf-8")

    def byte_to_variance_selector(self, byte: bytes) -> bytes:
        if byte < 16:
            return chr(0xFE00 + byte)
        else:
            return chr(0xE0100 + (byte - 16))

    def encode(self, sentence: str, base: str) -> str:
        encoded_bytes = sentence.encode("utf-8")
        return base + "".join(self.byte_to_variance_selector(b) for b in encoded_bytes)

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> int:
        reward = 0.0
        if answer is not None:
            try:
                if answer == entry["answer"]:
                    return 1.0
                elif len(answer) == len(entry["answer"]):
                    score = [1.0 if a == b else 0.0 for a, b in zip(answer, entry["answer"])]
                    reward = sum(score) / len(score)
                else:
                    reward = 0.01
            except:
                reward = 0.01
        return reward


class EmojiMysteryCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(EmojiMysteryCurriculum.__name__, EmojiMysteryConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="num_words_in_sentence",
                levels=[3, 10, 20, 35],
                description="Number of words in the sentence",
                lower_field_name="min_words_in_sentence",
                upper_field_name="max_words_in_sentence",
            ),
        )


register_dataset(DATASET_NAME, EmojiMysteryDataset, EmojiMysteryConfig, EmojiMysteryCurriculum)
