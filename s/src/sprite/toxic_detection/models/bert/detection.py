import collections
from typing import Dict, Iterable, List, Optional, Tuple, Union

from sprite.bricks.tokenizations.bert.tokenization import Tokenizer as BertTokenizer
from sprite.toxic_detection.modeling_utils import PreTrainedModel
from sprite.toxic_detection.models.bert.classification_model import Classifier
from sprite.utils.trie import Trie


class Detector(PreTrainedModel):
    """Toxic detector .

    Depending on `bert_tokenizer` and `bert_51m`.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    _WINDOW_SIZE = 4

    def __init__(
        self,
        *,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = {}
        self._config = config
        self._vocab_size = config.get("vocab_size", 50000)
        self._hidden_size = config.get("hidden_size", 512)
        self._num_heads = config.get("num_heads", 8)
        self._maxlen = config.get("maxlen", 512)
        self._n_layers = config.get("n_layers", 8)
        self._tag_num = config.get("tag_num", 6)

        self._tokenizer = BertTokenizer(maxlen=self._maxlen)
        self._classifier = Classifier(self.config)
        self._trie = Trie()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self._vocab_size = config.get("vocab_size", 50000)
        self._hidden_size = config.get("hidden_size", 512)
        self._num_heads = config.get("num_heads", 8)
        self._maxlen = config.get("maxlen", 512)
        self._n_layers = config.get("n_layers", 8)
        self._tag_num = config.get("tag_num", 6)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def classifier(self):
        return self._classifier

    @property
    def trie(self):
        return self._trie

    def score(self, text: str) -> List[float]:
        """Scoring the input text.

        Args:
            input (str):
                Text input.

        Returns:
            List[float]:
                The toxic score of the input .
        """
        input = self._tokenizer.encode_tensor(
            text, maxlen=self.config.get("maxlen", 512)
        ).view(1, -1)
        toxic_score = self._classifier.score(input)
        toxic_score = [round(s, 3) for s in toxic_score]
        return toxic_score

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

        if "config" not in model_file:
            raise KeyError("""`model_file` should include `config`!""")
        if not isinstance(model_file["config"], Dict):
            raise ValueError("""`config` should be a dict!""")

        config = model_file["config"]
        self.__init__(config=config)

        if "_classifier" not in model_file:
            raise KeyError("""`model_file` should include `_classifier`!""")
        if not isinstance(model_file["_classifier"], collections.OrderedDict):
            raise ValueError("""`_classifier` should be a OrderedDict!""")
        else:
            self._classifier.load_state_dict(model_file["_classifier"])
            self._classifier.eval()

        if "bert_tokenizer" in model_file:
            self._tokenizer.from_pretrained(model_file["bert_tokenizer"])
        else:
            self._tokenizer.from_pretrained("bert_tokenizer")

        if "trie" in model_file:
            self._trie.from_pretrained(model_file["trie"])

    def add_sensitive_words(self, words: Iterable[str]) -> None:
        for word in words:
            self._trie.add(word.lower())

    def sensitive_words_detect(self, text: str) -> List[Tuple[str, int, int]]:
        """sensitive_words_detect fot text (Multimode matching).

        Args:
            text (str):
            The text need to be multimode match.

        Returns:
            List[Tuple[str,int,int]]:
                Pattern as (match token,start,end).
                Where `match token` = `text`[`start`:`end`]
        """
        text = text.lower()
        return self._trie.multimode_match(text)

    def detect(self, text: str) -> Dict:
        """Detects toxic contents and sensitive words for `text`.

        Args:
            text (str):
            The text need to be detected.

        Returns:
            Dict:
                Pattern as {
                        "sensitive_words" :  List[Tuple[str, int, int]],
                        "toxic_score " : Dict[str,float]
                    }.
        """

        out = {
            "sensitive_words": self.sensitive_words_detect(text),
            "toxic_score": dict(
                zip(
                    [
                        "toxic",
                        "severe_toxic",
                        "obscene",
                        "threat",
                        "insult",
                        "identity_hate",
                    ],
                    self.score(text),
                )
            ),
        }
        return out
