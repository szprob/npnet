import random

import torch

from sprite.bricks.models.bert.bert import BERT


def test_bert():
    config = {
        "maxlen": 128,
        "n_layers": 2,
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_heads": 4,
    }

    model = BERT(config=config)

    inputs = torch.tensor(
        [
            [
                random.randint(0, config["vocab_size"] - 1)
                for _ in range(config["maxlen"])
            ],
            [
                random.randint(0, config["vocab_size"] - 1)
                for _ in range(config["maxlen"])
            ],
        ]
    )
    inputs = inputs.long()
    result = model(inputs)
    assert type(result) == torch.Tensor
