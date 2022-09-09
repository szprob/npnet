from sprite.spelling_correction.models.mlp.rank_block import RankModel, make_feature


def test_rank_block():
    model = RankModel()
    aim_word = "trike"
    sim_words = ["trikes", "tik"]
    m2c = {"chautauquas": ["chautauqua"]}
    cnts = {"daylight": 200, "trikes": 1000}
    left_words = ["<S>", "daylight"]
    right_words = ["<E>"]
    word2int = {
        "": 3,
        "<S>": 1,
        "<E>": 2,
        "daylight": 4,
        "trikes": 5,
    }
    cooccurance_map = {(4, 5): 1}

    feature = make_feature(
        aim_word,
        sim_words,
        m2c,
        cnts,
        left_words,
        right_words,
        word2int,
        cooccurance_map,
    )

    res = model.rank(feature)
    assert res.shape == (2,)
    assert res.dtype == "float32"

    aim_word = ""
    sim_words = ["daylightsss"]
    m2c = {}
    cnts = {}
    left_words = []
    right_words = []
    word2int = {
        "": 3,
        "<S>": 1,
        "<E>": 2,
        "daylight": 4,
        "trikes": 5,
    }
    cooccurance_map = {}

    feature = make_feature(
        aim_word,
        sim_words,
        m2c,
        cnts,
        left_words,
        right_words,
        word2int,
        cooccurance_map,
    )

    res = model.rank(feature)
    assert res.shape == (1,)
    assert res.dtype == "float32"
