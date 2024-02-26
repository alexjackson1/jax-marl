from typing import Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import chex

from ..multi_agent_env import MultiAgentEnv

from .shared import Activity, Config
from .logic import GameLogic
from .actions import ActionSpace
from .observations import ObservationSpace
from .state import State
from .logger import GameLogger


class TraitorsGame(MultiAgentEnv):
    """Traitors game environment suitable for JIT compilation with JAX."""

    NUM_ACTIVITIES = 4
    MAX_PHASES = 4
    MAX_PHASE_STEPS = 4

    def __init__(self, config: Config) -> None:
        """Construct a new TraitorsGame environment."""
        self.config = config
        self.logger = GameLogger("traitors")

        # Basic configuration
        self.num_agents = config.num_agents
        self.init_traitors = config.init_traitors
        self.num_days = config.num_days
        self.roundtables = config.roundtables
        self.secret_meetings = config.secret_meetings
        self.shields = config.shields
        self.num_symbols = config.num_symbols
        self.shield_success_rate = config.shield_success_rate

        # Determine sizes of action/observation spaces
        self.num_moves = self.count_moves()
        self.obs_size = self.count_obs_size()

        # Initialise action/observation spaces
        self.agent_ids = [i for i in range(self.num_agents)]

        self.action_spaces = {
            i: ActionSpace(self.num_agents, self.num_symbols, self.num_symbols)
            for i in self.agent_ids
        }

        self.observation_spaces = {
            i: ObservationSpace(
                self.num_agents,
                self.num_days,
                TraitorsGame.NUM_ACTIVITIES,
                TraitorsGame.MAX_PHASES,
                TraitorsGame.MAX_PHASE_STEPS,
                self.num_symbols,
                self.num_symbols,
            )
            for i in self.agent_ids
        }

    def count_moves(self) -> int:
        return sum(
            [
                1,  # NOOP
                2,  # End Game / Banish Again
                2,  # Eliminate / Recruit
                1,  # Shield
                self.num_agents,  # Vote
                self.num_agents * self.num_symbols,  # Open Signals
                self.num_agents * self.num_symbols,  # Secret Signals
            ]
        )

    def count_obs_size(self) -> int:
        return sum(
            [
                self.num_agents,  # Player id
                self.num_days,  # Day
                TraitorsGame.NUM_ACTIVITIES,  # Activities
                TraitorsGame.MAX_PHASES,  # Phase
                TraitorsGame.MAX_PHASE_STEPS,  # Phase step
                self.num_agents,  # Dead
                self.num_agents * 2,  # Shields (Doesn't have, does have)
                self.num_agents,  # At risk/tiebreak
                self.num_agents * self.num_agents * self.num_symbols,  # Open Signals
                self.num_agents * self.num_agents * self.num_symbols,  # Secret Signals
                self.num_agents * self.num_agents,  # Votes
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        state = State.create(key, self.config)
        return self.get_obs(state), state

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # state = state.clear_transient()
        if state.activity == Activity.BREAKFAST:
            state = GameLogic.run_breakfast(key, state, actions, self.logger)
        elif state.activity == Activity.CHALLENGE:
            state = GameLogic.run_challenge(key, state, actions, self.logger)
        elif state.activity == Activity.ROUNDTABLE:
            state = GameLogic.run_roundtable(key, state, actions, self.logger)
        elif state.activity == Activity.SECRET_MEETING:
            state = GameLogic.run_secret_meeting(key, state, actions, self.logger)
        elif state.activity == Activity.ENDGAME:
            state = GameLogic.run_endgame(key, state, actions, self.logger)
        else:
            raise ValueError("Invalid activity.")

        obs = self.get_obs(state)
        # TODO: Add rewards
        rewards = {a: 100 for a in range(self.num_agents)}
        dones = {a: False for a in range(self.num_agents)}
        dones["__all__"] = state.finished
        info = {}

        return obs, state, rewards, dones, info

    def get_mask(self, state: State) -> Dict[int, chex.Array]:
        """Applies mask function to state."""
        if state.activity == Activity.BREAKFAST:
            return GameLogic.get_breakfast_mask(state, self.action_spaces)
        elif state.activity == Activity.CHALLENGE:
            return GameLogic.get_challenge_mask(state, self.action_spaces)
        elif state.activity == Activity.ROUNDTABLE:
            return GameLogic.get_roundtable_mask(state, self.action_spaces)
        elif state.activity == Activity.SECRET_MEETING:
            return GameLogic.get_secret_meeting_mask(state, self.action_spaces)
        elif state.activity == Activity.ENDGAME:
            return GameLogic.get_endgame_mask(state, self.action_spaces)
        else:
            raise ValueError("Invalid activity.")

    def get_obs(self, state: State) -> Dict[int, chex.Array]:
        """Applies observation function to state."""

        @partial(jax.vmap, in_axes=[0, None])
        def _agent_obs(agent_id: int, state: State) -> chex.Array:
            """Generate individual agent's observation"""
            id = jax.nn.one_hot(agent_id, self.num_agents)

            # Time
            day = jax.nn.one_hot(state.day, self.num_days)
            activity = jax.nn.one_hot(state.activity, TraitorsGame.NUM_ACTIVITIES)
            phase = jax.nn.one_hot(state.phase, TraitorsGame.MAX_PHASES)
            phase_step = jax.nn.one_hot(state.phase_step, TraitorsGame.MAX_PHASE_STEPS)

            # Players
            dead = jnp.where((state.eliminated + state.banished) > 0, 1, 0)
            faithful = jnp.where(state.roles == 0, 1, 0)
            traitors = jnp.where(state.roles == 1, 1, 0)
            shields = jnp.where(state.has_shield == 1, 1, 0)
            at_risk = jnp.where(state.at_risk == 1, 1, 0)

            # Open signals
            open_signals = jnp.zeros(
                (self.num_symbols, self.num_agents, self.num_agents)
            ).flatten()
            votes = jnp.zeros((self.num_agents, self.num_agents)).flatten()

            # Hidden signals
            hidden_signals = jnp.zeros(
                (self.num_symbols, self.num_agents, self.num_agents)
            ).flatten()
            t_action_votes = jnp.zeros((2, self.num_agents)).flatten()
            t_target_votes = jnp.zeros((self.num_agents, self.num_agents)).flatten()

            # Endgame votes
            endgame_votes = jnp.zeros((2, self.num_agents)).flatten()

            # Full observation
            obs = jnp.concatenate(
                [
                    id,
                    day,
                    activity,
                    phase,
                    phase_step,
                    dead,
                    faithful,
                    traitors,
                    shields,
                    at_risk,
                    open_signals,
                    votes,
                    hidden_signals,
                    t_action_votes,
                    t_target_votes,
                    endgame_votes,
                ]
            )

            return obs

        obs = _agent_obs(jnp.arange(self.num_agents), state)
        return {i: obs[i] for i in range(self.num_agents)}

    @property
    def name(self) -> str:
        """Environment name."""
        return "Traitors"

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError


def example():
    key = jax.random.PRNGKey(0)

    from jaxmarl import make
    from .shared import uk_series_2

    env = make("traitors", config=uk_series_2())

    def _print_obs(obs):
        return jax.tree_util.tree_map(lambda x: x.shape, obs)

    def _print_state(state):
        return jax.tree_util.tree_map(
            lambda x: x.shape if "shape" in dir(x) else x, state
        )

    obs, state = env.reset(key)

    while state.day <= 1:
        print(state.timestep)

        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        # print("obs:", _print_obs(obs))
        # print("state:", _print_state(state))

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agent_ids)
        }

        # print("actions:", actions)

        # Perform the step transition.
        obs, state, reward, done, infos = env.step(key_step, state, actions)

        # print("reward:", reward[0])


if __name__ == "__main__":
    with jax.disable_jit():
        example()
