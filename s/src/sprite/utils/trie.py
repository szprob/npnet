from typing import Dict, Iterable, List, Optional, Tuple, Union

from sprite.module_utils import PreTrainedModule


class Trie(PreTrainedModule):
    """
    Trie in Python. Creates a Trie out of a list of words.

    Can be used for tokenization and multimode matching.

    Attributes:
        words (Optional[Iterable[str]] , optional):
            The words need to be added to trie.
            Defaults to None.
    """

    def __init__(self, words: Optional[Iterable[str]] = None) -> None:
        super().__init__()
        self.data = {}
        if words is not None:
            self.build_tree(words)

    def __repr__(self):
        return str(self.data)

    def _load(self, model: Union[str, Dict]) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (Union[str, Dict]):
                Model file need to be loaded.
                Can be either:
                    - A string, the path of a pretrained model.
                    - A state dict containing model weights.

        Raises:
            ValueError: model file should be a dict.
        """

        if isinstance(model, str):
            model_file = self._load_file(model)
        else:
            model_file = model

        if not isinstance(model_file, Dict):
            raise ValueError("""model file should be a dict!""")

        self.set_params(model_file)

    def set_params(self, model_file: Dict) -> None:
        """Set model params from `model_file`.

        Args:
            model_file (Dict):
                Dict containing model params.
        """

        if "data" in model_file:
            if isinstance(model_file["data"], Dict):
                self.data = model_file["data"]

    def build_tree(self, words: Iterable[str]):
        for word in words:
            self.add(word)

    def add(self, word: str) -> None:
        """Add `word` to `data` trie representation.

        Passes over every char (utf-8 char) on word and recursively
        adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.
        This function is idempotent,
        adding twice the same word will leave the trie unchanged

        """
        if not word:
            # Prevent empty string
            return
        ref = self.data
        ref["count"] = ref.get("count", 0) + 1
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
            ref["count"] = ref.get("count", 0) + 1
        ref["end"] = 1

    def query(self, word: str) -> bool:
        """Returns if the `word` is in the trie."""
        ref = self.data
        for w in word:
            if w not in ref:
                return False
            ref = ref[w]
        return ref.get("end", 0) == 1

    def prefix_check(self, prefix: str) -> bool:
        """If there's any word in the trie that starts with the given `prefix`."""
        ref = self.data
        for w in prefix:
            if w not in ref:
                return False
            ref = ref[w]
        return ref.get("count", 0) > 0

    def prefix_count(self, prefix: str) -> int:
        """Returns the count of word starts with the given `prefix`."""
        ref = self.data
        for w in prefix:
            if w not in ref:
                return 0
            ref = ref[w]
        return ref["count"]

    def delete(self, word: str) -> None:
        """Delete `word` form data`."""
        if not self.query(word):
            return

        ref = self.data
        ref["count"] = ref.get("count", 0) - 1
        for w in word:
            ref = ref[w]
            ref["count"] = ref.get("count", 0) - 1
        ref["end"] = 0

    def _get_key(self, prefix: str, ref: Dict) -> List[str]:
        word_list = []
        if ref.get("end", 0) == 1:
            word_list.append(prefix)
        for w in ref.keys():
            if w != "count" and w != "end":
                word_list.extend(self._get_key(prefix + w, ref[w]))
        return word_list

    def get_prefix(self, prefix: str) -> List[str]:
        """Returns words started with prefix."""

        if not self.prefix_check(prefix):
            return []
        ref = self.data
        for w in prefix:
            ref = ref[w]
        return self._get_key(prefix, ref)

    def multimode_match(self, text: str) -> List[Tuple[str, int, int]]:
        """Multimode matching for text

        Args:
            text (str):
            The text need to be multimode match.

        Returns:
            List[Tuple[str,int,int]]:
                Pattern as (match token,start,end).
                Where `match token` = `text`[`start`:`end`]
        """

        out = []
        length = len(text)
        start = 0
        while start < length:
            ref = self.data
            for end in range(start, length):
                char = text[end]
                # cur_prefix not in tree
                if char not in ref:
                    if len(out) == 0 or start != out[-1][1]:
                        start += 1
                    else:
                        start = out[-1][2]
                    break

                ref = ref[char]
                # cur_prefix in tree
                if ref.get("end", 0) == 1:
                    if len(out) == 0 or start != out[-1][1]:
                        out.append((text[start : end + 1], start, end + 1))
                    else:
                        out[-1] = (text[start : end + 1], start, end + 1)
                    if end == length - 1:
                        start = end
                        break

            else:
                start += 1

        return out
