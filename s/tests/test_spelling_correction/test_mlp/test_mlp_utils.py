# from pytest_mock import mocker

from sprite.spelling_correction.models.mlp import counting_utils
from sprite.spelling_correction.models.mlp.correction_utils import (
    word2int_diff,
    word_cnts,
)


def test_counting_utils():
    # lcs
    assert counting_utils.lcs(s="", t="aa") == 0
    assert counting_utils.lcs(t="a", s="aa") == 1
    assert counting_utils.lcs("", "") == 0

    # lcs ratio
    assert counting_utils.lcs_ratio(s="", t="aa") < 0.00001
    assert counting_utils.lcs_ratio("", "") < 0.00001
    assert counting_utils.lcs_ratio("a", "aa") < 0.50001
    assert counting_utils.lcs_ratio("a", "aa") > 0.49999

    # get cooccurance
    corpus = [["i", "like", "you"], ["i", "love", "you"]]
    word2int = {"i": 1, "like": 2, "love": 3, "you": 4, "<S>": 5, "<E>": 6}
    cooccurance = counting_utils.get_cooccurance(
        corpus=corpus, word2int=word2int, window_size=4, num_workers=2
    )
    assert cooccurance[(1, 2)] == 1

    cooccurance2 = counting_utils.get_cooccurance(
        corpus=corpus, word2int=word2int, window_size=4, num_workers=1
    )
    for key in cooccurance:
        assert cooccurance[key] == cooccurance2[key]


def test_correction_utils():
    # _word_cnts
    corpus = [["a", "b"], ["bb", "b", "c"]]
    cnts = word_cnts(corpus)
    assert cnts["bb"] == 1

    # _update_word2int
    word2int = {"a": 0, "b": 1}
    words = ["b", "c"]
    new_dict = word2int_diff(word2int, words)
    assert new_dict["c"] == 2
    assert "a" not in new_dict.keys()

    new_dict = word2int_diff(word2int, ["a"])
    assert len(new_dict) == 0
