from enum import Enum, IntEnum
from typing import Union
import chex

from flax import struct
import numpy as np


@struct.dataclass
class Config:
    """Configuration for the game."""

    # Basic configuration
    num_agents: int
    num_symbols: int = 3
    num_days: int = 12
    init_traitors: int = 3

    # Activities
    roundtables: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: np.zeros(True, dtype=int)
    )
    secret_meetings: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: np.zeros(True, dtype=int)
    )

    # Shields
    shields: chex.Array = struct.field(
        pytree_node=False, default_factory=lambda: np.zeros(12, dtype=int)
    )
    shield_success_rate: float = 0.25

    # Communication
    open_length: int = 3
    hidden_length: int = 2

    def __repr__(self):
        s = f"Config(num_agents={self.num_agents}, num_symbols={self.num_symbols}, "
        s += f"num_days={self.num_days}, init_traitors={self.init_traitors}, "
        s += f"rountables={self.roundtables.shape}, secret_meetings={self.secret_meetings.shape}, "
        s += f"shields={self.shields.shape}, shield_success_rate={self.shield_success_rate}, "
        s += f"open_length={self.open_length}, hidden_length={self.hidden_length})"
        return s


class Activity(IntEnum):
    """Enumeration of possible activities in the game."""

    BREAKFAST = 0
    CHALLENGE = 1
    DISCUSSION = 2
    ROUNDTABLE = 3
    SECRET_MEETING = 4
    ENDGAME = 5


class BreakfastPhase(IntEnum):
    GROUP_DISCUSSION = 1


class ChallengePhase(IntEnum):
    ATTEMPT_SHIELD = 1


class RoundtablePhase(IntEnum):
    GROUP_DISCUSSION = 1
    ROUNDTABLE_VOTE = 2


class SecretMeetingPhase(IntEnum):
    TRAITORS_DISCUSSION = 1
    TRAITORS_ACTION_VOTE = 2
    TRAITORS_TARGET_VOTE = 3
    RECRUITMENT_OFFER = 4


class EndgamePhase(IntEnum):
    ENDGAME_VOTE = 1
    ENDGAME_BANISHMENT = 2


Phase = Union[
    BreakfastPhase, ChallengePhase, RoundtablePhase, SecretMeetingPhase, EndgamePhase
]


def uk_series_2():
    return Config(
        num_agents=22,
        roundtables=np.array([False] + [True] * 10),
        secret_meetings=np.array([False] + [True] * 9 + [False]),
        shields=np.array([3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]),
    )
