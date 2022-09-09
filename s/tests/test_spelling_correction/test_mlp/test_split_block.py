from sprite.spelling_correction.models.mlp.split_block import WordSpliter


def test_word_split():
    model = WordSpliter()
    assert model.split("freefire") == ["freefire"]
    model.update_vocab({"free", "fire"})
    assert model.split("freefire") == ["free", "fire"]
    text = "freefire" * 3
    assert model.split(text) == ["free", "fire"] * 3
    text = "freefire" * 100
    assert model.split(text) == [text]
