from sprite.utils.trie import Trie


def test_trie():
    words = ["free fire", "fire", "free", "bmg"]
    t = Trie(words)

    # query
    assert t.query("free")
    assert not t.query("fre")
    assert not t.query("")

    # get_prefix
    assert t.get_prefix("") == ["free", "free fire", "fire", "bmg"]
    assert t.get_prefix("free") == ["free", "free fire"]
    assert t.get_prefix("b") == ["bmg"]
    assert t.get_prefix("dfadfafdas") == []

    # prefix_check
    assert t.prefix_check("free")
    assert t.prefix_check("fre")
    assert t.prefix_check("")

    # prefix count
    assert t.prefix_count("free") == 2
    assert t.prefix_count("") == 4

    # add same
    t.add("free")
    t.add("free")
    assert t.prefix_count("free") == 4
    assert t.prefix_count("") == 6
    t.add("free")
    assert t.prefix_check("free")
    assert t.prefix_count("free") == 5
    assert t.prefix_count("") == 7

    # delete
    t.delete("free")
    assert not t.query("free")
    assert t.prefix_check("free")
    assert t.prefix_count("free") == 4
    t.delete("")
    assert not t.query("")

    # multimode_match
    words = ["free fire", "fire", "free", "bmg"]
    t = Trie(words)
    text = "i like free fire,but i hate bmg!"
    assert t.multimode_match(text) == [("free fire", 7, 16), ("bmg", 28, 31)]
    text = "i like free fir,but i hate bmg !"
    assert t.multimode_match(text) == [("free", 7, 11), ("bmg", 27, 30)]
    text = ""
    assert t.multimode_match(text) == []
    text = " "
    assert t.multimode_match(text) == []
