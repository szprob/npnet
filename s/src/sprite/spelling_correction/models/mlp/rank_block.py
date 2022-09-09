import itertools
from typing import Dict, Iterable, Tuple

import Levenshtein
import numpy as np
import torch
from metaphone import doublemetaphone
from torch import nn

from sprite.spelling_correction.models.mlp import counting_utils


def _get_cooccurance_sum(
    sim_int: int,
    context_ints: Iterable[int],
    cooccurance_map: Dict[Tuple[int, int], int],
) -> float:
    """Get sum of cooccurance between `sim_int` and all `context_ints`"""
    if sim_int == 0:
        return 0
    return sum(
        [
            cooccurance_map.get(
                (min(sim_int, context_int), max(sim_int, context_int)), 0
            )
            for context_int in context_ints
        ]
    )


def _calculate_cooc_feature(
    left_words: Iterable[str],
    right_words: Iterable[str],
    sim_words: Iterable[str],
    cnts: Dict[str, int],
    word2int: Dict[str, int],
    cooccurance_map: Dict[Tuple[int, int], int],
) -> np.ndarray:
    """Calculate cooccurance divided by occurance of `sim_words`."""

    # center word occurance
    sim_occurance = np.array([cnts.get(w, 0) + 1 for w in sim_words])

    # map2int
    context_ints = [
        word2int.get(w, 0) for w in itertools.chain(left_words, right_words)
    ]
    sim_ints = [word2int.get(w, 0) for w in sim_words]

    # ngram
    cooccurance_sum = np.array(
        [_get_cooccurance_sum(s, context_ints, cooccurance_map) for s in sim_ints]
    )
    score = np.power(cooccurance_sum, 2 / 5) / np.power(sim_occurance, 2 / 5)

    return score


def _calculate_occ_feature(
    sim_words: Iterable[str],
    cnts: Dict[str, int],
) -> np.ndarray:
    """Performs counts based feature function on sim_words."""
    return np.array([(cnts.get(w, 0) ** 0.1 / 6) for w in sim_words])


def _calculate_ortho_feature(aim_word: str, sim_words: Iterable[str]) -> np.ndarray:
    """Performs orthographic based feature function on sim_words."""
    return np.array([Levenshtein.jaro(aim_word, s) for s in sim_words])


def _calculate_phonetic_feature(aim_word: str, sim_words: Iterable[str]) -> np.ndarray:
    """Performs phonetic based feature function on sim_words."""
    aim_phonetic = doublemetaphone(aim_word)[0]
    return np.array(
        [Levenshtein.jaro(aim_phonetic, doublemetaphone(s)[0]) for s in sim_words]
    )


def _calculate_miss_feature(
    aim_word: str, sim_words: Iterable[str], m2c: Dict[str, Iterable[str]]
) -> np.ndarray:
    """Checks if the sim_words in miss_words."""
    c_words = set(m2c.get(aim_word, set()))
    return np.array([float(s in c_words) for s in sim_words])


def _calculate_lcs_feature(aim_word: str, sim_words: Iterable[str]) -> np.ndarray:
    """Performs lcs based feature function on sim_words."""
    return np.array([counting_utils.lcs_ratio(aim_word, w) for w in sim_words])


def make_feature(
    aim_word: str,
    sim_words: Iterable[str],
    m2c: Dict[str, Iterable[str]],
    cnts: Dict[str, int],
    left_words: Iterable[str],
    right_words: Iterable[str],
    word2int: Dict[str, int],
    cooccurance_map: Dict[Tuple[int, int], int],
) -> np.ndarray:
    """Make features for rank model.

    Args:
        aim_word (str):
            Aim wrong word.
        sim_words (Iterable[str]):
            Sim candidates.
        m2c (Dict[str, Iterable[str]]):
            Mapping of missspelled2correct pre collated.
        cnts (Dict[str, int]):
            Word counts.
        left_words (Iterable[str]):
            Left context words.
        right_words (Iterable[str]):
            Right context words.
        word2int (Dict[str,int]):
            Mapping dict from word to idx.
        cooccurance_map (Dict[Tuple[int, int], int]):
            Dict stores cooccurance numbers.

    Returns:
        np.ndarray:
            Score for sim words.
    """

    x = np.concatenate(
        [
            _calculate_cooc_feature(
                left_words, right_words, sim_words, cnts, word2int, cooccurance_map
            ).reshape([-1, 1]),
            _calculate_occ_feature(sim_words, cnts).reshape([-1, 1]),
            _calculate_ortho_feature(aim_word, sim_words).reshape([-1, 1]),
            _calculate_phonetic_feature(aim_word, sim_words).reshape([-1, 1]),
            _calculate_miss_feature(aim_word, sim_words, m2c).reshape([-1, 1]),
            _calculate_lcs_feature(aim_word, sim_words).reshape([-1, 1]),
        ],
        axis=1,
    ).astype("float32")
    return x


class RankModel(nn.Module):
    """Minimally-supervised linear model for correction."""

    def __init__(self, input_dim: int = 6) -> None:
        """Create a rank model for correction.

        Args:
            input_dim (int, optional):
                Input features dim.
                Defaults to 6.
        """
        super().__init__()
        self._input_dim = input_dim
        self._fc_block = nn.Sequential(
            nn.Linear(self._input_dim, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (b,input_dim)
        x = self._fc_block(x)
        x = x.view(-1)
        # x (b)
        return x

    @torch.no_grad()
    def rank(self, features: np.ndarray) -> np.ndarray:
        """Rank sim words by features.

        Args:
            features (np.array):
                Features for rank model.

        Returns:
            np.array:
                Score for sim words.
        """
        out = self.forward(torch.tensor(features).float())
        return torch.sigmoid(out).detach().cpu().numpy()
