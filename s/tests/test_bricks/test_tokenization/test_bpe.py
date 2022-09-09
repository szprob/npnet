from sprite.bricks.tokenizations.bert.bpe import BPE


def test_bpe():
    vocab = {
        "lov": 1,
        "##ing": 2,
        "aaaaaaaaaaaaaa": 3,
        "word": 4,
    }
    model = BPE(vocab=vocab, max_token_length=10, unk_token="[UNK]")
    assert model.tokenize("loving") == ["lov", "##ing"]
    assert model.tokenize("aaaaaaaaaaaaaa") == ["[UNK]"]
    assert model.tokenize("word") == ["word"]
    assert model.tokenize("loved") == ["[UNK]"]
    assert model.tokenize("") == []
