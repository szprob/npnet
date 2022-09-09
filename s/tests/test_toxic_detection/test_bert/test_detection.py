# from pytest_mock import mocker
import pickle
from typing import List

from sprite.toxic_detection.models.bert.detection import Detector as BertToxicDetector


def test_detector(mocker):

    model = BertToxicDetector()
    # mock model
    mock_value = {
        "config": {
            "vocab_size": 50000,
            "hidden_size": 512,
            "num_heads": 8,
            "maxlen": 512,
            "n_layers": 8,
            "tag_num": 6,
        },
        "_classifier": model._classifier.state_dict(),
        "bert_tokenizer": {
            "vocab": {
                "work": 1,
                "##ing": 2,
                "[UNK]": 3,
                "[SEP]": 4,
                "[PAD]": 5,
                "[CLS]": 6,
                "[MASK]": 7,
            }
        },
    }
    with open(f"{model._tmpdir.name}/toxic_detector_bert_51m.pkl", "wb") as file:
        pickle.dump(mock_value, file)

    model._hdfs_download = mocker.patch.object(BertToxicDetector, "_hdfs_download")

    # init model
    model.from_pretrained("toxic_detector_bert_51m")

    # test classifier
    text = "i like free fire"
    res = model.score(text)
    assert len(res) == mock_value["config"]["tag_num"]
    assert isinstance(res, List)
    text = ""
    res = model.score(text)
    assert len(res) == mock_value["config"]["tag_num"]
    assert isinstance(res, List)
    text = "fasfsefewfewfwefw" * 1000
    res = model.score(text)
    assert len(res) == mock_value["config"]["tag_num"]
    assert isinstance(res, List)
    text = "   "
    res = model.score(text)
    assert len(res) == mock_value["config"]["tag_num"]
    assert isinstance(res, List)
    text = "$%^$&%^(*^&#:}{:}><:"
    res = model.score(text)
    assert len(res) == mock_value["config"]["tag_num"]
    assert isinstance(res, List)

    # test detect
    model.add_sensitive_words(["free", "fire", "free fire"])

    text = "i like free fire"
    res = model.detect(text)
    assert res["sensitive_words"] == [("free fire", 7, 16)]
    assert "toxic_score" in res
    assert isinstance(res["toxic_score"]["toxic"], float)

    text = ""
    res = model.detect(text)
    assert res["sensitive_words"] == []
    assert len(list(res["toxic_score"])) == mock_value["config"]["tag_num"]
    assert isinstance(res["toxic_score"]["obscene"], float)

    text = "$%^$&%^(*^&#:}{:}><:"
    res = model.detect(text)
    assert res["sensitive_words"] == []
    assert len(list(res["toxic_score"])) == mock_value["config"]["tag_num"]
    assert isinstance(res["toxic_score"]["obscene"], float)

    text = "     "
    res = model.detect(text)
    assert res["sensitive_words"] == []
    assert len(list(res["toxic_score"])) == mock_value["config"]["tag_num"]
    assert isinstance(res["toxic_score"]["obscene"], float)
