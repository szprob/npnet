import collections
from typing import Dict, Optional, Union

import torch
from torch import nn

from sprite.bricks.models.bert.embedding import Embeddings
from sprite.bricks.models.bert.encoder import Encoder
from sprite.bricks.models.modeling_utils import PreTrainedModel
from sprite.bricks.models.nn_utils import get_pad_mask


class BERT(PreTrainedModel, nn.Module):

    """Modeling bert.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        PreTrainedModel.__init__(self)
        nn.Module.__init__(self)

        if config is None:
            config = {}
        self.config = config

        self.vocab_size = config.get("vocab_size", 50000)
        self.hidden_size = config.get("hidden_size", 512)
        self.num_heads = config.get("num_heads", 8)
        self.maxlen = config.get("maxlen", 512)
        self.n_layers = config.get("n_layers", 8)

        self.embed = Embeddings(
            self.vocab_size, maxlen=self.maxlen, hidden_size=self.hidden_size
        )
        self.encoders = Encoder(
            d_model=self.hidden_size, num_heads=self.num_heads, n_layers=self.n_layers
        )

        # Can also have linear and tanh here
        # This implemention has no nsp block.
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, inputs: torch.Tensor, segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of bert.

        Args:
            inputs (torch.Tensor):
                The index of words. shape:(b,l)
            segment_ids (Optional[torch.Tensor], optional):
                The index of segments.
                This arg is not usually used.
                Defaults to None.

        Returns:
            torch.Tensor:
                BERT result. shape:(b,l,d)
        """
        x = self.embed(inputs, segment_ids)
        attn_mask = get_pad_mask(inputs)
        x = self.encoders(x, attn_mask)
        # can have h_pooled hera: fc(x[:,0])
        return x

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

        if "config" in model_file:
            if not isinstance(model_file["config"], Dict):
                raise ValueError("""`config` should be a dict!""")

        config = model_file["config"]

        if "vocab_size" not in config:
            raise KeyError("""`config` should include `vocab_size`!""")

        if "hidden_size" not in config:
            raise KeyError("""`config` should include `hidden_size`!""")

        if "maxlen" not in config:
            raise KeyError("""`config` should include `maxlen`!""")

        if "num_heads" not in config:
            raise KeyError("""`config` should include `num_heads`!""")

        if "n_layers" not in config:
            raise KeyError("""`config` should include `n_layers`!""")

        self.__init__(config=model_file["config"])

        if "state_dict" in model_file:
            if not isinstance(model_file["state_dict"], collections.OrderedDict):
                raise ValueError("""`state_dict` should be a OrderedDict!""")
            else:
                self.load_state_dict(model_file["state_dict"])
