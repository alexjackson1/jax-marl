from typing import Dict, TYPE_CHECKING
import chex

if TYPE_CHECKING:
    from .interface import State

from .shared import Activity, BreakfastPhase, ChallengePhase


class GameLogic:

    @staticmethod
    def run_breakfast(
        key: chex.PRNGKey, state: "State", actions: Dict[str, chex.Array]
    ) -> "State":
        LENGTH = 2
        assert state.activity == Activity.BREAKFAST
        assert state.phase == BreakfastPhase.GROUP_DISCUSSION
        assert state.phase_step <= LENGTH

        if state.phase_step == 1:
            # logger.heading_2("Breakfast")

            state = state.reset_shields()
            state = state.discover_death()

        # TODO
        self.group_discussion()

        state = state.step()

    @staticmethod
    def run_challenge(
        key: chex.PRNGKey, state: "State", actions: Dict[str, chex.Array]
    ):
        pass

    @staticmethod
    def run_roundtable(
        key: chex.PRNGKey, state: "State", actions: Dict[str, chex.Array]
    ):
        pass

    @staticmethod
    def run_secret_meeting(
        key: chex.PRNGKey, state: "State", actions: Dict[str, chex.Array]
    ):
        pass

    @staticmethod
    def run_endgame(key: chex.PRNGKey, state: "State", actions: Dict[str, chex.Array]):
        pass
