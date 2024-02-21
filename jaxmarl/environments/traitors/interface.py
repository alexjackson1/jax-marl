from typing import Dict, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp

import chex
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete

from .shared import Activity, ChallengePhase, Config, Phase
from .schedule import next_timestep
from .logic import GameLogic


@struct.dataclass
class State:
    config: Config

    # Timestep
    day: int
    activity: Activity
    phase: Phase
    phase_step: int
    finished: bool

    # Player Properties
    players: chex.Array  # id of player
    traitors: chex.Array  # 1 if traitor
    murdered: chex.Array  # 1 if murdered
    banished: chex.Array  # 1 if banished
    at_risk: chex.Array  # 1 if at risk
    has_shield: chex.Array  # 1 if has shield

    # Game state
    last_death: Optional[int]  # id of last death

    banishment_selection: Optional[int]  # Group banishment target
    traitor_action_selection: Optional[int]  # Traitor action
    traitor_target_selection: Optional[int]  # Traitor recruitment/murder target
    banish_again: Optional[bool]  # Whether to banish again in the endgame

    def step(self) -> "State":
        ts = next_timestep(self, self.config)
        return self.replace(
            day=ts.day,
            activity=ts.activity,
            phase=ts.phase,
            phase_step=ts.phase_step,
            finished=ts.finished,
        )

    def reset_shields(self) -> "State":
        return self.replace(has_shield=jnp.zeros_like(self.has_shield))

    def discover_death(self) -> "State":
        if self.last_death is not None:
            new_murdered = jnp.where(
                self.murdered == 1 or self.players == self.last_death, 1, 0
            )
            return self.replace(murdered=new_murdered, last_death=None)

        return self


class TraitorsGame(MultiAgentEnv):
    """Traitors game environment suitable for JIT compilation with JAX."""

    NUM_ACTIVITIES = 4
    MAX_PHASES = 4
    MAX_PHASE_STEPS = 4

    def __init__(self, config: Config) -> None:
        """Construct a new TraitorsGame environment."""
        self.config = config

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
        self.action_set = jnp.arange(self.num_moves)
        self.action_spaces = {i: Discrete(self.num_moves) for i in self.agent_ids}
        self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agent_ids}

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
        traitor_idxs = jax.random.choice(
            key, self.num_agents, shape=(self.init_traitors,), replace=False
        )
        traitors = jax.nn.one_hot(traitor_idxs, self.num_agents).sum(axis=0)

        state = State(
            config=self.config,
            day=0,
            activity=Activity.CHALLENGE,
            phase=ChallengePhase.ATTEMPT_SHIELD,
            phase_step=1,
            finished=False,
            players=jnp.arange(self.num_agents),
            traitors=traitors,
            murdered=jnp.zeros(self.num_agents),
            banished=jnp.zeros(self.num_agents),
            at_risk=jnp.zeros(self.num_agents),
            has_shield=jnp.zeros(self.num_agents),
            last_death=jnp.zeros(self.num_agents),
            traitor_target_selection=None,
            traitor_action_selection=None,
            banishment_selection=None,
            banish_again=None,
        )

        return self.get_obs(state), state

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
        # if self.is_first_activity_of_day:
        #     logger.heading_1(self.day)
        #     logger.info("There are %d players remaining.", len(self.players))

        if state.activity == Activity.BREAKFAST:
            GameLogic.run_breakfast(key, state, actions)
        elif state.activity == Activity.CHALLENGE:
            GameLogic.run_challenge(key, state, actions)
        elif state.activity == Activity.ROUNDTABLE:
            GameLogic.run_roundtable(key, state, actions)
        elif state.activity == Activity.SECRET_MEETING:
            GameLogic.run_secret_meeting(key, state, actions)
        elif state.activity == Activity.ENDGAME:
            GameLogic.run_endgame(key, state, actions)
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
            activity = jax.nn.one_hot(state.activity.value, TraitorsGame.NUM_ACTIVITIES)
            phase = jax.nn.one_hot(state.phase.value, TraitorsGame.MAX_PHASES)
            phase_step = jax.nn.one_hot(state.phase_step, TraitorsGame.MAX_PHASE_STEPS)

            # Players
            dead = jnp.where((state.murdered + state.banished) > 0, 1, 0)
            faithful = jnp.where(state.traitors == 0, 1, 0)
            traitors = jnp.where(state.traitors == 1, 1, 0)
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
    key = jax.random.PRNGKey(0)

    from jaxmarl import make
    from .shared import uk_series_2

    env = make("traitors", config=uk_series_2())

    obs, state = env.reset(key)
    print(jax.tree_util.tree_map(lambda x: x.shape, obs))
    print(jax.tree_util.tree_map(lambda x: x.shape if "shape" in dir(x) else x, state))


if __name__ == "__main__":
    with jax.disable_jit():
        example()
