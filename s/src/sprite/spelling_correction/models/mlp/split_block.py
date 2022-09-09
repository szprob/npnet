from typing import Iterable, List, Optional, Set

import wordninja


class WordSpliter:
    """A spliter for long words."""

    def __init__(
        self,
        vocab: Optional[Iterable[str]] = None,
    ) -> None:

        """Create a spliter.

        Args:
            vocab (Optional[Iterable[str]]  , optional):
                Vocab for spliter to look up.
                None vocab means no split on words.
                Defaults to None.

        """
        if vocab is None:
            vocab = set()
        self._vocab = set(vocab)

    def split(self, word: str, max_word_length: int = 100) -> List[str]:
        """Main split function for long word.

        Args:
            word (str):
                Long word for split.
            max_word_length (int, optional):
                Max length of word for split.
                Words more than this value will not be split.
                Defaults to 100.

        Returns:
            List[str]:
                Words after split.
        """

        if len(word) > max_word_length:
            return [word]

        split_words = wordninja.split(word)

        # filter condition
        all_in_vocab = all((w in self._vocab for w in split_words))
        all_greater_1 = all((len(w) > 1 for w in split_words))

        if all_in_vocab and all_greater_1:
            return split_words
        else:
            return [word]

    def update_vocab(self, new_vocab: Set[str]) -> None:
        self._vocab.update(new_vocab)
