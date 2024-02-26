from collections import OrderedDict
from enum import Enum
from typing import Dict, Sequence, Tuple

import chex
import jax

from ..spaces import Dict as DictSpace, Discrete, MultiDiscrete, Space


class EndgameAction(Enum):
    """Enumeration of possible endgame actions."""

    BANISH_AGAIN = 0
    END_GAME = 1


class TraitorAction(Enum):
    """Enumeration of possible traitor actions."""

    RECRUIT = 0
    ELIMINATE = 1


class ChallengeAction(Enum):
    """Enumeration of possible shield actions."""

    ATTEMPT = 1


class ActionSpace(DictSpace):
    """Action space for the game."""

    def __init__(self, num_agents: int, num_open_symbols: int, num_hidden_symbols: int):
        super().__init__(
            {
                "challenge": Discrete(2),
                "player_vote": Discrete(num_agents),
                "traitor_action_vote": Discrete(2),
                "endgame_action_vote": Discrete(2),
                "open_signals": MultiDiscrete([num_open_symbols, num_agents]),
                "hidden_signals": MultiDiscrete([num_hidden_symbols, num_agents]),
            }
        )


import jax.numpy as jnp
