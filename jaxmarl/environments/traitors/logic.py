from typing import Dict, TYPE_CHECKING, List, OrderedDict
import chex
import jax

if TYPE_CHECKING:
    from .interface import State

from .shared import Activity, BreakfastPhase, ChallengePhase, RoundtablePhase
from .logger import GameLogger

from .actions import ActionSpace

import jax.numpy as jnp


class GameLogic:
    @staticmethod
    def group_discussion(state: "State", actions: Dict[int, chex.Array]) -> "State":
        # for agent_id, agent_action in actions.items():
        #     assert agent_action.shape == ()

        #     idx = 1 + 2 + 2 + 1 + state.config.num_agents
        #     idx_2 = idx + state.config.num_agents * state.config.num_symbols

        #     if agent_action >= idx and agent_action < idx_2:
        #         target_id = (agent_action - idx) // state.config.num_symbols
        #         symbol = (agent_action - idx) % state.config.num_symbols
        #         state = state.open_signal(agent_id, target_id, symbol)
        #     else:
        #         print("Invalid action")
        #         pass

        return state

    @staticmethod
    def run_breakfast(
        _key: chex.PRNGKey,
        state: "State",
        actions: Dict[int, chex.Array],
        logger: GameLogger,
    ) -> "State":
        """
        Run the breakfast activity.

        - First, the game will reset the shields and announce the death of any
          murdered players.
        - Then, each step, the players can signal publicly using a finite
          number of symbols referring to other players.

        Phases:
            Group Discussion: `open_length` steps.

        Args:
            key: The PRNG key (unused).
            state: The current state of the game.
            actions: The actions taken by the players.

        Returns:
            The new state of the game.
        """
        assert state.activity == Activity.BREAKFAST
        assert state.phase == BreakfastPhase.GROUP_DISCUSSION
        assert state.phase_step <= state.config.open_length

        if state.phase_step == 1:
            logger.heading_2("Breakfast")

            state = state.reset_shields()
            state = state.discover_death()

        state = GameLogic.group_discussion(state, actions)
        state = state.step()

        return state

    @staticmethod
    def run_challenge(
        key: chex.PRNGKey,
        state: "State",
        actions: Dict[int, chex.Array],
        logger: GameLogger,
    ) -> "State":
        """
        Run the challenge activity.

        - Each player has the opportunity to attempt to secure one of a limited
          number of shields.
        - A shield will protect the player from being murdered by the traitors.
        - Once all the players have decided whether to attempt to secure a
          shield, the game will allocate the shields to successful players.
        - It is possible for no players to secure a shield.

        Phases:
            Attempt Shield: 1 step.

        Args:
            key: The PRNG key.
            state: The current state of the game.
            actions: The actions taken by the players.

        Returns:
            The new state of the game.
        """
        assert state.activity == Activity.CHALLENGE
        assert state.phase == ChallengePhase.ATTEMPT_SHIELD
        assert state.phase_step == 1

        if state.config.shields[state.day] == 0:
            logger.info("No shields available for the challenge.")
            state = state.step()

        logger.heading_2("Challenge")
        state = GameLogic.attempt_shields(state, actions, logger)
        state = GameLogic.allocate_shield(key, state, logger)
        state = state.step()

        return state

    @staticmethod
    def run_roundtable(
        key: chex.PRNGKey,
        state: "State",
        actions: Dict[int, chex.Array],
        logger: GameLogger,
    ) -> "State":
        """
        Run the roundtable activity.

        - The players will discuss and vote on a player to banish from the game.
        - If the players fail to reach a majority, the players with the most
          votes will be put at risk of banishment.
        - The remaining players will vote for one of the at-risk players to be
          banished from the game.
        - If the players fail to reach a majority again, the game will randomly
          select one of the at-risk players to banish.

        Phases:
            Group Discussion: `open_length` steps.
            Roundtable Vote: 2 steps.

        Args:
            key: The PRNG key.
            state: The current state of the game.
            actions: The actions taken by the players.

        Returns:
            The new state of the game.
        """

        assert state.activity == Activity.ROUNDTABLE

        if not state.roundtable_today:
            state = state.step()
            return state

        if state.phase == RoundtablePhase.GROUP_DISCUSSION:
            LENGTH = 3
            assert state.phase_step <= LENGTH

            if state.phase_step == 1:
                logger.heading_2("Roundtable")

            state = GameLogic.group_discussion(state, actions)

        elif state.phase == RoundtablePhase.ROUNDTABLE_VOTE:
            assert state.phase_step <= 2

            if state.phase_step == 1:
                result = GameLogic.banishment_vote(state.players)
                if isinstance(result, int):
                    state = state.banish(result)
                else:
                    # logger.info("The players failed to reach a majority.")
                    state = state.set_at_risk(result)

            elif state.phase_step == 2:
                assert state.at_risk is not None
                result = GameLogic.banishment_vote(state.at_risk)
                if isinstance(result, int):
                    state = state.banish(result)
                else:
                    # logger.info("The players failed to reach a majority.")
                    players = [p for p in self.players.values() if p.id in result]
                    # TODO: shoudl be alive players (and above...)
                    self.banish(self.generator.choice(state.players))

                state = state.clear_at_risk()

            else:
                raise ValueError("Invalid phase step for roundtable vote.")
        else:
            raise ValueError("Invalid phase for roundtable activity.")

        state = state.step()
        return state

    @staticmethod
    def run_secret_meeting(
        key: chex.PRNGKey,
        state: "State",
        actions: Dict[int, chex.Array],
        logger: GameLogger,
    ) -> "State":
        return state.step()

    @staticmethod
    def attempt_shields(
        state: "State",
        actions: Dict[int, OrderedDict[str, chex.Array]],
        logger: GameLogger,
    ) -> "State":
        for sender_id, action in actions.items():
            assert "challenge" in action
            assert action["challenge"].shape == ()
            assert action["challenge"] in [0, 1]

            if action == 1:
                logger.info(f"Player {sender_id} is attempting to secure a shield.")
                state = state.attempt_shield(sender_id)

        return state

    @staticmethod
    def allocate_shield(
        key: chex.PRNGKey, state: "State", logger: GameLogger
    ) -> "State":
        if sum(state.attempted_shield) == 0:
            logger.info("No players attempted to secure a shield.")
            return state

        candidates = state.shield_attempters()
        for shield in range(1, state.num_shields + 1):
            inner_key = jax.random.fold_in(key, shield)
            key_1, key_2 = jax.random.split(inner_key)
            success: List[int] = []
            for canidate in candidates:
                # TODO: Fold key
                if jax.random.uniform(key_1) < state.config.shield_success_rate:
                    success.append(canidate)

            if len(success) == 0:
                logger.info(f"No player secured shield {shield}.")
                continue

            elif len(success) > 1:
                logger.debug(f"Multiple players secured shield {shield}.")
                success = [jax.random.choice(key_2, success)]

            logger.info(f"Player {success[0].id} secured shield {shield}.")
            candidates.remove(success[0])
            state = state.allocate_shield(success[0])

        return state

    def banishment_vote(players: List[int]) -> List[int]:
        pass

    @staticmethod
    def get_breakfast_mask(state: "State", spaces: Dict[int, ActionSpace]):
        assert state.activity == Activity.BREAKFAST
        assert state.phase == BreakfastPhase.GROUP_DISCUSSION
        assert state.phase_step <= state.config.open_length

        return {
            k: {
                "no_op": jnp.ones((1,)),
                "challenge": jnp.zeros((1,)),
                "player_vote": jnp.zeros((state.config.num_agents,)),
                "traitor_action_vote": jnp.zeros((2,)),
                "endgame_action_vote": jnp.zeros((2,)),
                "open_signals": jnp.ones(
                    (state.config.open_length, state.config.num_agents)
                ),
                "hidden_signals": jnp.zeros(
                    (state.config.hidden_length, state.config.num_agents)
                ),
            }
            for k in spaces
        }

    @staticmethod
    def get_challenge_mask(state: "State", spaces: Dict[int, ActionSpace]):
        assert state.activity == Activity.BREAKFAST
        assert state.phase == BreakfastPhase.GROUP_DISCUSSION
        assert state.phase_step <= state.config.open_length

        return {
            k: {
                "no_op": jnp.ones((1,)),
                "challenge": jnp.ones((1,)),
                "player_vote": jnp.zeros((state.config.num_agents,)),
                "traitor_action_vote": jnp.zeros((2,)),
                "endgame_action_vote": jnp.zeros((2,)),
                "open_signals": jnp.zeros(
                    (state.config.open_length, state.config.num_agents)
                ),
                "hidden_signals": jnp.zeros(
                    (state.config.hidden_length, state.config.num_agents)
                ),
            }
            for k in spaces
        }

    @staticmethod
    def get_roundtable_mask(state: "State", spaces: Dict[int, ActionSpace]):
        pass

    @staticmethod
    def get_secret_meeting_mask(state: "State", spaces: Dict[int, ActionSpace]):
        pass

    @staticmethod
    def get_endgame_mask(state: "State", spaces: Dict[int, ActionSpace]):
        pass
