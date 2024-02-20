import jax
import jax.numpy as jnp
from typing import Dict
import chex
from functools import partial
from flax import struct
from typing import Tuple

import numpy as np

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete


@struct.dataclass
class State:
    done: chex.Array
    step: int


class TraitorsGame(MultiAgentEnv):
    """Traitors game environment suitable for JIT compilation with JAX."""

    def __init__(
        self,
        num_agents: int,
        init_traitors: int = 3,
        num_days: int = 12,
        roundtables: chex.Array = np.array([False] + [True] * 11),
        secret_meetings: chex.Array = np.array([True] * 9 + [False, True, False]),
        shields: chex.Array = np.array([3] * 3 + [2] * 3 + [1] * 3 + [0] * 3),
        shield_success_rate: float = 0.25,
        num_symbols: int = 3,
    ) -> None:
        """
        Construct a new TraitorsGame environment.

        Args:
            num_agents (int): maximum number of agents within the environment,
            used to set array dimensions.
        """
        self.num_agents = num_agents

        self.init_traitors = init_traitors
        self.num_days = num_days
        self.roundtables = roundtables
        self.secret_meetings = secret_meetings
        self.shields = shields
        self.num_symbols = num_symbols
        self.shield_success_rate = shield_success_rate

        self.num_moves = sum(
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

        NUM_ACIVITIES = 4
        MAX_PHASES = 4
        MAX_PHASE_STEPS = 4
        self.obs_size = sum(
            self.num_agents,  # Player id
            self.num_days,  # Day
            NUM_ACIVITIES,  # Activity
            MAX_PHASES,  # Phase
            MAX_PHASE_STEPS,  # Phase step
            self.num_agents,  # Dead
            self.num_agents * 2,  # Shields (Doesn't have, does have)
            self.num_agents,  # At risk/tiebreak
            self.num_agents * self.num_agents * self.num_symbols,  # Open Signals
            self.num_agents * self.num_agents * self.num_symbols,  # Secret Signals
            self.num_agents * self.num_agents,  # Votes
        )

        self.action_set = jnp.arange(self.num_moves)
        self.action_spaces = {
            i: Discrete(self.num_moves) for i in range(self.num_agents)
        }
        self.observation_spaces = {
            i: Discrete(self.obs_size) for i in range(self.num_agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""

        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError


def example():
    num_agents = 5
    key = jax.random.PRNGKey(0)

    from jaxmarl import make

    env = make("switch_riddle", num_agents=num_agents)

    obs, state = env.reset(key)
    env.render(state)

    for _ in range(20):
        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        env.render(state)
        print("obs:", obs)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        print(
            "action:",
            env.game_actions_idx[actions[env.agents[state.agent_in_room]].item()],
        )

        # Perform the step transition.
        obs, state, reward, done, infos = env.step(key_step, state, actions)

        print("reward:", reward["agent_0"])


if __name__ == "__main__":
    with jax.disable_jit():
        example()
