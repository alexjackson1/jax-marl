from typing import List, Tuple, Dict, Literal, OrderedDict

from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jr

import chex
from flax import struct


AgentID = str
SpaceID = Literal["target"]

Actions = Dict[AgentID, OrderedDict[SpaceID, chex.Array]]


class Phase(IntEnum):
    """Phase of the game."""

    TOWN_DISCUSSION = 0
    PLAYER_BANISHED = 1
    WOLF_DISCUSSION = 2
    PLAYER_IS_EATEN = 3

    @staticmethod
    def next(phase: "Phase") -> "Phase":
        return (phase + 1) % 4

    @staticmethod
    def label(phase: "Phase") -> str:
        return [
            "Town Discussion",
            "Player Banished",
            "Wolf Discussion",
            "Player Eaten",
        ][phase]


class GameStatus(IntEnum):
    """Status of the game."""

    TIMEOUT = -1
    NONE = 0
    VILLAGER_WIN = 1
    WOLF_WIN = 2

    @staticmethod
    def label(game_status: "GameStatus") -> str:
        return ["None", "Villagers", "Wolves", "Timeout"][game_status]


@struct.dataclass
class State:
    """State of the environment."""

    # Game information
    phase: Phase
    day: int
    finished: bool
    game_status: chex.Array

    # Player information
    order: chex.Array  # order[agent] = order of the agent
    status: chex.Array  # 1: alive, 0: dead
    role: chex.Array  # 1: wolf, 0: villager
    votes: chex.Array  # votes[agent, target]
    targets: chex.Array  # targets[agent, target]

    # Statistics
    self_votes: chex.Array
    kill_dead: chex.Array
    cannibalism: chex.Array
    accords: chex.Array

    def create(ids: chex.Array, roles: chex.Array, max_days: int) -> "State":
        """Creates a new state."""
        num_agents = len(ids)
        return State(
            phase=Phase.WOLF_DISCUSSION,
            day=0,
            finished=False,
            game_status=GameStatus.NONE,
            order=ids,
            status=jnp.ones_like(ids),
            role=roles,
            votes=jnp.zeros((num_agents, num_agents)),
            targets=jnp.zeros((num_agents, num_agents)),
            self_votes=jnp.zeros((max_days, 4)),
            kill_dead=jnp.zeros((max_days, 4)),
            cannibalism=jnp.zeros((max_days, 4)),
            accords=jnp.zeros((max_days, 4)),
        )


@chex.dataclass
class GameConfig:
    num_phases: int = 4  # fixed
    max_day: int = 10
    num_agents: int = 8
    num_wolves: int = 2


@chex.dataclass
class RewardConfig:
    """Reward configuration for the Werewolf environment."""

    victory: float = +25.0
    """Reward for winning the game."""
    loss: float = -25.0
    """Punishment for losing the game."""

    death: float = -1.0
    """Punishment for dying."""
    kill: float = 1.0
    """Reward for killing."""
    execute_dead: float = -1.0
    """Punishment for executing a dead player."""
    kill_wolf: float = -1.0
    """Punishment for killing a wolf."""


class WerewolfLogic:
    """Game logic for the Werewolf environment."""

    game_config: GameConfig
    rewards_config: RewardConfig

    def __init__(
        self,
        game_config: GameConfig,
        rewards_config: RewardConfig,
        agents: List[AgentID],
    ):
        """Initialises the game logic."""
        self.game_config = game_config
        self.rewards_config = rewards_config
        self.agents = agents

    @property
    def num_agents(self) -> int:
        """Returns the number of agents in the environment."""
        return self.game_config.num_agents

    # Pre-Action Processing ==============================================================
    def pre_actions(
        self, state: State, actions: Actions, rewards: chex.Array
    ) -> Tuple[State, Actions, chex.Array]:
        """Updates the state to reflect the action choices before execution."""
        # TODO: Agents were punished here
        # TODO: Previous targets were updated here

        # Update wolf targets or everyone's votes
        for agent, action in actions.items():
            idx = self.agents.index(agent)
            state = lax.cond(
                (state.phase == Phase.PLAYER_BANISHED)
                | (state.phase == Phase.TOWN_DISCUSSION),
                lambda _: WerewolfLogic.update_vote(state, idx, action["target"]),
                lambda _: WerewolfLogic.update_target(state, idx, action["target"]),
                operand=None,
            )

        # TODO: Difference and influence were applied here
        # TODO: Flexibility was applied here

        return state, actions, rewards

    @staticmethod
    def update_vote(state: State, idx: int, target: chex.Array) -> State:
        """Updates the target of the agent."""
        chex.assert_shape(target, ())
        chex.assert_scalar(idx)
        agent = state.order[idx]
        return state.replace(votes=state.votes.at[agent, target].set(1))

    @staticmethod
    def update_target(state: State, idx: int, target: chex.Array) -> State:
        """Updates the target of the wolf agent."""
        chex.assert_shape(target, ())
        chex.assert_scalar(idx)
        agent = state.order[idx]
        return state.replace(targets=state.targets.at[agent, target].set(1))

    # Action Execution ===================================================================
    def run_actions(
        self, key: chex.PRNGKey, state: State, actions: Actions, rewards: chex.Array
    ) -> Tuple[State, Actions, chex.Array]:
        """Execute the actions and update the state."""
        # Banishment
        state, actions, rewards = lax.cond(
            (state.phase == Phase.PLAYER_BANISHED),
            lambda _: self.step_town(key, state, actions, rewards),
            lambda _: (state, actions, rewards),
            operand=None,
        )

        # Eaten by wolves
        state, actions, rewards = lax.cond(
            (state.phase == Phase.PLAYER_IS_EATEN),
            lambda _: self.step_wolf(key, state, actions, rewards),
            lambda _: (state, actions, rewards),
            operand=None,
        )

        return state, actions, rewards

    def step_town(
        self, key: chex.PRNGKey, state: State, actions: Actions, rewards: chex.Array
    ) -> Tuple[State, Actions, chex.Array]:
        """Executes the town actions."""
        # Conduct a majority vote to determine the target
        target_idx = self.majority_vote(key, state.votes)

        # Update the number of agents voting for themselves
        sv_count = jnp.trace(state.votes)
        state = WerewolfLogic.update_self_vote(state, sv_count)

        # TODO: Accord was applied here.

        # Construct traced booleans if target is alive/dead
        target_alive = (state.status[target_idx]).astype(bool)
        target_dead = (1 - state.status[target_idx]).astype(bool)

        # Kill the target by setting status to 0
        state = WerewolfLogic.update_status(state, target_idx, 0)

        # If target was alive, punish the killed agent
        oh_target = jax.nn.one_hot(target_idx, self.num_agents)
        rewards = rewards + target_alive * oh_target * self.rewards_config.death

        # If target was dead, punish others and update metrics
        rewards = rewards + target_dead * self.rewards_config.execute_dead
        state = WerewolfLogic.update_kill_dead(state, target_alive)

        return state, actions, rewards

    def step_wolf(
        self, key: chex.PRNGKey, state: State, actions: Actions, rewards: chex.Array
    ) -> Tuple[State, Actions, chex.Array]:
        """Executes the wolf actions."""
        # Conduct a majority votes of the wolves to determine the target
        wolf_tm = state.targets * jnp.reshape(state.role, (-1, 1))
        target_idx = self.majority_vote(key, wolf_tm)

        # Update the number of wolves voting for themselves
        sv_count = jnp.trace(wolf_tm)  # sum of the diagonal
        state = WerewolfLogic.update_self_vote(state, sv_count)

        # TODO: Accord was applied here.

        # Construct traced booleans if target is alive/dead/wolf
        target_alive = (state.status[target_idx]).astype(bool)
        target_dead = (1 - state.status[target_idx]).astype(bool)
        target_wolf = (state.role[target_idx]).astype(bool)

        # Kill the target by setting status to 0
        state = WerewolfLogic.update_status(state, target_idx, 0)

        # If target was alive, punish the dead villager and reward wolves
        v_punish = target_alive * (1 - state.role) * self.rewards_config.death
        w_reward = target_alive * state.role * self.rewards_config.kill
        rewards = rewards + v_punish + w_reward

        # If target was dead, punish the wolves and update metrics
        w_punish_1 = target_dead * state.role * self.rewards_config.execute_dead
        state = WerewolfLogic.update_kill_dead(state, target_alive)
        rewards = rewards + w_punish_1

        # If target was a wolf, punish the wolves and update metrics
        w_punish_2 = target_wolf * state.role * self.rewards_config.kill_wolf
        rewards = rewards + w_punish_2
        state = WerewolfLogic.update_cannibalism(state, target_wolf)

        return state, actions, rewards

    def majority_vote(self, key: chex.PRNGKey, target_matrix: chex.Array) -> chex.Array:
        """Conducts a majority vote to determine the target for village/wolves."""
        chex.assert_shape(target_matrix, (self.num_agents, self.num_agents))
        targets = jnp.sum(target_matrix, axis=0)
        at_risk = (targets == jnp.max(targets)).astype(int)
        return jnp.argmax(at_risk * jr.uniform(key, at_risk.shape))

    @staticmethod
    def update_status(state: State, target: chex.Array, status: chex.Array) -> State:
        """Updates the status of the target."""
        chex.assert_shape([target, status], ())
        return state.replace(status=state.status.at[target].set(status))

    @staticmethod
    def update_self_vote(state: State, count: chex.Array) -> State:
        """Updates the self-vote statistics."""
        chex.assert_shape([count], ())
        day, phase = state.day, state.phase
        return state.replace(self_votes=state.self_votes.at[day, phase].set(count))

    @staticmethod
    def update_kill_dead(state: State, dead: chex.Array) -> State:
        """Updates the kill-dead statistics."""
        chex.assert_shape([dead], ())
        day, phase = state.day, state.phase
        return state.replace(kill_dead=state.kill_dead.at[day, phase].set(dead))

    @staticmethod
    def update_cannibalism(state: State, wolf: chex.Array) -> State:
        """Updates the cannibalism statistics."""
        chex.assert_shape(wolf, ())
        day, phase = state.day, state.phase
        return state.replace(cannibalism=state.cannibalism.at[day, phase].set(wolf))

    # Post-Action Processing =============================================================
    def post_actions(
        self, state: State, rewards: chex.Array
    ) -> Tuple[State, chex.Array]:
        """Updates the state to reflect any win conditions or time elapse."""
        # Compute the number of wolves and villagers
        wolves = jnp.sum(state.role * state.status)
        villagers = jnp.sum(state.status) - wolves

        # Check if the wolves have won
        state, rewards = lax.cond(
            (wolves >= villagers),
            lambda _: self.set_wolf_win(state, rewards),
            lambda _: (state, rewards),
            operand=None,
        )

        # Check if the villagers have won
        state, rewards = lax.cond(
            (wolves == 0),
            lambda _: self.set_villager_win(state, rewards),
            lambda _: (state, rewards),
            operand=None,
        )

        # Check if the time has elapsed
        state, rewards = lax.cond(
            (state.day >= self.game_config.max_day - 1),
            lambda _: self.set_timeout(state, rewards),
            lambda _: (state, rewards),
            operand=None,
        )

        return state, rewards

    def set_wolf_win(
        self, state: State, rewards: chex.Array
    ) -> Tuple[State, chex.Array]:
        """Sets the game status to finished via wolf victory."""
        state = state.replace(finished=True, game_status=GameStatus.WOLF_WIN)

        rewards = rewards + state.role * self.rewards_config.victory
        rewards = rewards + (1 - state.role) * self.rewards_config.loss

        return state, rewards

    def set_villager_win(
        self, state: State, rewards: chex.Array
    ) -> Tuple[State, chex.Array]:
        """Sets the game status to finished via villager victory."""
        state = state.replace(finished=True, game_status=GameStatus.VILLAGER_WIN)

        rewards = rewards + (1 - state.role) * self.rewards_config.victory
        rewards = rewards + state.role * self.rewards_config.loss

        return state, rewards

    def set_timeout(
        self, state: State, rewards: chex.Array
    ) -> Tuple[State, chex.Array]:
        """Sets the game status to finished via timeout."""
        state = state.replace(finished=True, game_status=GameStatus.TIMEOUT)
        rewards = rewards + self.rewards_config.loss
        return state, rewards

    # Time Step ==========================================================================
    def step_time(self, state: State) -> State:
        """Updates the time of the environment by one step."""
        new_phase = (state.phase + 1) % self.game_config.num_phases
        new_day = state.day + (new_phase == 0)
        return state.replace(phase=new_phase, day=new_day)
