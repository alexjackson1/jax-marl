from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import chex
from flax import struct

from .shared import Activity, ChallengePhase, Config, Phase
from .schedule import next_timestep


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
    roles: chex.Array  # 1 if traitor

    eliminated: chex.Array  # 1 if murdered
    banished: chex.Array  # 1 if banished

    has_shield: chex.Array  # 1 if has shield
    attempted_shield: chex.Array  # 1 if attempted shield

    at_risk: chex.Array  # 1 if at risk

    # Game state
    last_death: Optional[int]  # id of last death
    banishment_selection: Optional[int]  # Group banishment target
    traitor_action_selection: Optional[int]  # Traitor action
    traitor_target_selection: Optional[int]  # Traitor recruitment/murder target
    banish_again: Optional[bool]  # Whether to banish again in the endgame

    # Communication
    open_signals: Optional[chex.Array]
    hidden_signals: Optional[chex.Array]

    @property
    def timestep(self):
        return (self.day, self.activity, self.phase, self.phase_step)

    @staticmethod
    def assign_traitors(key: chex.PRNGKey, num_agents: int, count: int) -> chex.Array:
        """
        Selects traitors and returns one hot array.

        Args:
            key: random number generator key
            num_agents: total number of agents
            count: traitor count

        Returns:
            One-hot array of traitor positions.
        """
        indices = jax.random.choice(key, num_agents, shape=(count,), replace=False)
        return jax.nn.one_hot(indices, num_agents).sum(axis=0)

    @staticmethod
    def create(key: chex.PRNGKey, config: Config) -> "State":
        return State(
            config=config,
            # Timestep
            day=0,
            activity=Activity.CHALLENGE,
            phase=ChallengePhase.ATTEMPT_SHIELD,
            phase_step=1,
            finished=False,
            # Player Properties
            roles=State.assign_traitors(key, config.num_agents, config.init_traitors),
            eliminated=jnp.zeros(config.num_agents),
            banished=jnp.zeros(config.num_agents),
            at_risk=jnp.zeros(config.num_agents),
            has_shield=jnp.zeros(config.num_agents),
            attempted_shield=jnp.zeros(config.num_agents),
            # Game state
            last_death=jnp.zeros(config.num_agents),
            traitor_target_selection=None,
            traitor_action_selection=None,
            banishment_selection=None,
            banish_again=None,
            # Communication
            open_signals=None,
            hidden_signals=None,
        )

    def step(self) -> "State":
        """Advance the state by one timestep."""
        ts = next_timestep(self, self.config)
        return self.replace(
            day=ts.day,
            activity=ts.activity,
            phase=ts.phase,
            phase_step=ts.phase_step,
            finished=ts.finished,
        )

    # VOTING METHODS
    @property
    def players_at_risk(self) -> Tuple[int, ...]:
        """Players who are at risk."""
        return jnp.where(self.at_risk == 1)

    def set_at_risk(self, at_risk: chex.Array) -> "State":
        return self.replace(
            at_risk=jax.nn.one_hot(at_risk, self.config.num_agents).sum(0)
        )

    def clear_at_risk(self) -> "State":
        return self.replace(at_risk=jnp.zeros(self.config.num_agents))

    # COMMUNICATION METHODS
    def open_signal(self, sender_idx: int, content_idx: int, symbol: int) -> "State":
        """Sends an open signal from sender of the form `<symbol>(<content_idx>)`."""
        return self.replace(
            open_signals=self.open_signals.at[sender_idx, content_idx, symbol].set(1)
        )

    def hidden_signal(self, sender_idx: int, content_idx: int, symbol: int) -> "State":
        """Sends a traitor-only signal from sender of the form `<symbol>(<content_idx>)`."""
        return self.replace(
            hidden_signals=self.hidden_signals.at[sender_idx, content_idx, symbol].set(
                1
            )
        )

    # BREAKFAST ACTIVITY METHODS
    def discover_death(self) -> "State":
        """Discover the last death and update the state."""
        if self.last_death is not None:
            id = self.last_death
            updated = jnp.where(
                (self.eliminated == 1) | (jnp.arange(self.config.num_agents) == id),
                1,
                0,
            )
            return self.replace(eliminated=updated, last_death=None)

        return self

    # CHALLENGE ACTIVITY METHODS
    @property
    def num_shields(self) -> int:
        """Number of shields available for the day."""
        return self.config.shields[self.day]

    @property
    def shield_attempters(self) -> Tuple[int, ...]:
        """Agents who attempted to shield."""
        return jnp.where(self.attempted_shield == 1)

    def attempt_shield(self, idx: int) -> "State":
        """Update state to reflect agent attempting to obtain a shield."""
        return self.replace(attempted_shield=self.attempted_shield.at[idx].set(1))

    def allocate_shield(self, idx: int) -> "State":
        """Allocate a shield to an agent."""
        return self.replace(has_shield=self.has_shield.at[idx].set(1))

    def reset_shields(self) -> "State":
        """Reset shield state for the next day."""
        return self.replace(
            has_shield=jnp.zeros_like(self.has_shield),
            attempted_shield=jnp.zeros_like(self.attempted_shield),
        )

    # ROUNDTABLE ACTIVITY METHODS
    @property
    def roundtable_today(self) -> bool:
        """Whether there is a roundtable today."""
        return self.config.roundtables[self.day]

    def banish(self, idx: int) -> "State":
        """Banish a player from the game."""
        return self.replace(banished=self.banished.at[idx].set(1))

    # SECRET MEETING ACTIVITY METHODS
    def eliminate(self, idx: int) -> "State":
        """Eliminate a player from the game."""
        return self.replace(last_death=idx)
