from sprite import MLPCorrector
from sprite.spelling_correction.trainers.mlp.trainer import Trainer


def test_trainer():
    model = MLPCorrector()
    t = Trainer(model)
    corpus = ["i like free fire", "we are students!", "i buy a lot of things"] * 100
    words = [
        ["i", "am", "poor", "stundent"],
        [
            "i",
            "don",
            "'",
            "t",
            "have",
            "money",
            "for",
            "by",
            "the",
            "cacharacter",
            "that",
            "'",
            "s",
            "wise",
            "thank",
            "you",
        ],
    ]
    labels = [
        ["<correct>", "<correct>", "<correct>", "student"],
        [
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
            "buy",
            "<correct>",
            "character",
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
            "<correct>",
        ],
    ]

    t.train(corpus, words, labels, num_workers=2, batch_size=4, max_step=100)
