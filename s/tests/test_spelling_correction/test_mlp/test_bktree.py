from sprite.spelling_correction.models.mlp import bktree


def test_bktree():
    # Levenshtein
    assert bktree.edit_distance("", "love") == 4
    assert bktree.edit_distance("ab", "ba") == 1
    assert bktree.edit_distance("", "") == 0
    assert bktree.edit_distance("play", "plya") == 1
    assert bktree.edit_distance("a", "b") == 1
    assert bktree.edit_distance("play", "clay") == 1

    # build bktree
    words = ["i", "like", "free", "fire", "", "", "", "free", "free"]
    t = bktree.BKTree(words)
    # query
    assert "free" in t.query("fre", 3)
    assert "free" in t.query("fre", 1)
    assert "free" in t.query("free", 0)
    assert len(t.query("free", 0)) == 1
    assert "i" in t.query("", 1)
    assert "i" in t.query(" ", 1)
    assert "" in t.query(" ", 1)
    assert "" not in t.query("  ", 1)
    # add
    t.add("bmg")
    assert "bmg" in t.query("bgm", 1)
    assert "" not in t.query("bgm", 1)
    assert set(["i", ""]) == set(t.query("", 1))
    t.add("bmg")
    t.add("bmg")
    t.add("bmg")
    assert len(t.query("bgm", 1)) == 1
    # batch query
    assert "bmg" in t.batch_query(["bgm", "bg"], 1, 2)[0]
    assert t.batch_query([], 1, 2) == []
    result = t.batch_query(["bgm", "bgm"], 1, 2)
    assert set(result[0]) == set(result[1])
