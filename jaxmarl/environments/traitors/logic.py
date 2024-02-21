from typing import Dict, TYPE_CHECKING
import chex

if TYPE_CHECKING:
    from .interface import State

from .shared import Activity, BreakfastPhase, ChallengePhase


class GameLogic:

    @staticmethod
    def run_breakfast(
        _key: chex.PRNGKey, state: "State", actions: Dict[int, chex.Array]
    ) -> "State":
        assert state.activity == Activity.BREAKFAST
        assert state.phase == BreakfastPhase.GROUP_DISCUSSION
        assert state.phase_step <= state.config.open_length

        if state.phase_step == 1:
            # logger.heading_2("Breakfast")

            state = state.reset_shields()
            state = state.discover_death()

        GameLogic.group_discussion(actions)

        state = state.step()

    @staticmethod
    def group_discussion(
        state: "State", actions: Dict[int, chex.Array]
    ) -> Dict[int, chex.Array]:
        # TODO: this is really wrong atm
        for id, action in actions.items():
            assert action.shape == ()

            idx = 1 + 2 + 2 + 1 + state.config.num_agents
            idx_2 = idx + state.config.num_agents * state.config.num_symbols
            if action >= idx and action < idx_2:
                # valid, update
                pass
            else:
                # invalid, ignore
                pass

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
