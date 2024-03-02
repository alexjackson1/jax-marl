from enum import IntEnum
from functools import partial
from typing import Dict, List, Literal, OrderedDict, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import chex
from flax import struct

from .render import WerewolfRender

from ..multi_agent_env import MultiAgentEnv
from ..spaces import Discrete, Dict as DictSpace


AgentID = int
SpaceID = Literal["target"]

Actions = Dict[AgentID, OrderedDict[SpaceID, chex.Array]]


class Phase(IntEnum):
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


class Winner(IntEnum):
    TIMEOUT = -1
    NONE = 0
    VILLAGERS = 1
    WOLVES = 2

    @staticmethod
    def label(winner: "Winner") -> str:
        return ["None", "Villagers", "Wolves", "Timeout"][winner]


@struct.dataclass
class State:
    """State of the environment."""

    # Game information
    phase: Phase
    day: int
    finished: bool
    winner: chex.Array

    # Player information
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
            winner=Winner.NONE,
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
class RewardConfig:
    victory: float = +25.0
    loss: float = -25.0

    death: float = -1.0
    kill: float = 1.0
    execute_dead: float = -1.0
    kill_wolf: float = -1.0


class WerewolfGame(MultiAgentEnv):
    """Jittable abstract base class for all jaxmarl Environments."""

    NUM_PHASES = 4
    MAX_DAY = 10

    num_agents: int
    num_wolves: int
    num_villagers: int

    agents: List[AgentID]
    observation_spaces: Dict[AgentID, chex.Array]
    action_spaces: Dict[AgentID, DictSpace]

    _renderer: WerewolfRender = None
    _reward_config: RewardConfig = RewardConfig()

    # Initialisation and configuration =========================================
    def __init__(self, num_agents: int, num_wolves: int):
        self.num_agents = num_agents
        self.num_wolves = num_wolves
        self.num_villagers = num_agents - num_wolves

        self.agents = WerewolfGame.init_agents(num_agents)
        self.observation_spaces = WerewolfGame.init_observation_spaces(self.agents)
        self.action_spaces = WerewolfGame.init_action_spaces(self.agents)

        self._renderer = WerewolfRender(self.num_agents)

    @staticmethod
    def init_agents(num_agents: int) -> List[AgentID]:
        """Initialises agents."""
        return [i for i in range(num_agents)]

    @staticmethod
    def init_observation_spaces(agents: List[AgentID]) -> Dict[AgentID, chex.Array]:
        """Initialises observation spaces."""
        obs_size = sum(
            [
                WerewolfGame.MAX_DAY + WerewolfGame.NUM_PHASES,  # one hot time
                1 + len(agents) * 4,  # own id (1), status, role, votes, targets
            ]
        )
        return {id: Discrete(obs_size) for id in agents}

    @staticmethod
    def init_action_spaces(agents: List[AgentID]) -> Dict[AgentID, DictSpace]:
        """Initialises action spaces."""
        return {id: DictSpace({"target": Discrete(len(agents))}) for id in agents}

    @staticmethod
    def assign_roles(key: chex.PRNGKey, counts: Tuple[int, int]) -> chex.Array:
        """Assigns roles to agents."""
        wc, vc = counts
        roles = jnp.concatenate([jnp.full((wc), 1), jnp.full((vc), 0)])
        return jr.permutation(key, roles)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[AgentID, chex.Array], State]:
        """Performs resetting of the environment."""
        id_shuff, role_shuff = jr.split(key, 2)
        counts = (self.num_wolves, self.num_villagers)
        ids = jr.permutation(id_shuff, self.num_agents)
        roles = WerewolfGame.assign_roles(role_shuff, counts)
        state = State.create(ids, roles, WerewolfGame.MAX_DAY)
        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state)

    # Step and transition functions ============================================
    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Actions
    ) -> Tuple[Dict[str, chex.Array], State, chex.Array, Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # Initialise the reward array and info dict
        rewards, infos = jnp.zeros(self.num_agents), {}

        def f1(key, state, actions, rewards):
            # Update the state w.r.t. the actions
            state, actions, rewards = self.update_data(state, actions, rewards)
            state, actions, rewards = self.run_actions(key, state, actions, rewards)
            state = self._next_phase(state)
            return state, actions, rewards, False

        def f2(state, actions, rewards):
            return state, actions, rewards, True

        state, actions, rewards, is_done = lax.cond(
            state.finished,
            lambda _: f2(state, actions, rewards),
            lambda _: f1(key, state, actions, rewards),
            operand=None,
        )

        state, rewards = self.check_end_conditions(state, rewards)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            lax.stop_gradient(rewards),
            {"__all__": is_done},
            infos,
        )

    def update_data(
        self, state: State, actions: Actions, rewards: chex.Array
    ) -> Tuple[State, Actions, chex.Array]:
        """Updates the votes/targets of the agents based on their actions."""
        # TODO: Agents were punished here
        # TODO: Previous targets were updated here

        # Update wolf targets or everyone's votes
        for agent, action in actions.items():
            state = lax.cond(
                (state.phase == Phase.PLAYER_BANISHED)
                | (state.phase == Phase.TOWN_DISCUSSION),
                lambda _: self._update_vote(state, agent, action["target"]),
                lambda _: self._update_target(state, agent, action["target"]),
                operand=None,
            )

        # TODO: Difference and influence were applied here
        # TODO: Flexibility was applied here

        return state, actions, rewards

    def run_actions(
        self, key: chex.PRNGKey, state: State, actions: Actions, rewards: chex.Array
    ) -> Tuple[State, Actions, chex.Array]:
        """Processes the deaths of the agents."""
        # Banishment
        state, actions, rewards = lax.cond(
            (state.phase == Phase.PLAYER_BANISHED),
            lambda _: self.run_town_actions(key, state, actions, rewards),
            lambda _: self._no_op(key, state, actions, rewards),
            operand=None,
        )

        # Eaten by wolves
        state, actions, rewards = lax.cond(
            (state.phase == Phase.PLAYER_IS_EATEN),
            lambda _: self.run_wolf_actions(key, state, actions, rewards),
            lambda _: self._no_op(key, state, actions, rewards),
            operand=None,
        )

        return state, actions, rewards

    def run_town_actions(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[int, Dict[str, int]],
        rewards: chex.Array,
    ) -> Tuple[State, Dict[int, Dict[str, int]], chex.Array]:
        target_idx = self._compute_target(key, state.votes)

        # Update the number of agents voting for themselves
        self_vote_count = jnp.trace(state.votes)
        state = self._update_self_votes(state, state.day, state.phase, self_vote_count)

        # TODO: Accord?

        # Construct traced booleans if target is alive/dead
        b_target_alive = (state.status[target_idx]).astype(bool)
        b_target_dead = (1 - state.status[target_idx]).astype(bool)

        # Kill the target by setting status to 0
        state = self._update_status(state, target_idx, 0)

        # If target was alive, punish the killed agent
        oh_target = jax.nn.one_hot(target_idx, self.num_agents)
        rewards = rewards + b_target_alive * oh_target * self._reward_config.death

        # If target was dead, punish other and update metrics
        rewards = rewards + b_target_dead * self._reward_config.execute_dead
        state = self._update_kill_dead(state, state.day, state.phase, b_target_alive)

        return state, actions, rewards

    def run_wolf_actions(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[int, Dict[str, int]],
        rewards: chex.Array,
    ) -> Tuple[State, Dict[int, Dict[str, int]], Dict[int, int]]:
        # Compute the target matrix using just the wolf votes
        wolf_tm = state.targets * jnp.reshape(state.role, (-1, 1))

        # Determine the target of the wolves
        target_idx = self._compute_target(key, wolf_tm)

        # Update the number of wolves voting for themselves
        self_vote_count = jnp.trace(wolf_tm)  # sum of the diagonal
        state = self._update_self_votes(state, state.day, state.phase, self_vote_count)

        # TODO: "Target Accord" was applied here but currently not implemented.

        # Construct traced booleans if target is alive/dead/wolf
        b_target_alive = (state.status[target_idx]).astype(bool)
        b_target_dead = (1 - state.status[target_idx]).astype(bool)
        b_target_wolf = (state.role[target_idx]).astype(bool)

        # Kill the target by setting status to 0
        state = self._update_status(state, target_idx, 0)

        # If target was alive, punish the villager and reward wolves
        v_punish = b_target_alive * (1 - state.role) * self._reward_config.death
        w_reward = b_target_alive * state.role * self._reward_config.kill
        rewards = rewards + v_punish + w_reward

        # If target was dead, punish the wolves and update metrics
        w_punish_1 = b_target_dead * state.role * self._reward_config.execute_dead
        state = self._update_kill_dead(state, state.day, state.phase, b_target_alive)
        rewards = rewards + w_punish_1

        # If target was a wolf, punish the wolves and update metrics
        w_punish_2 = b_target_wolf * state.role * self._reward_config.kill_wolf
        rewards = rewards + w_punish_2
        state = self._update_cannibalism(state, state.day, state.phase, b_target_wolf)

        return state, actions, rewards

    def check_end_conditions(
        self, state: State, rewards: Dict[AgentID, float]
    ) -> Tuple[State, chex.Array]:
        # Compute the number of wolves and villagers
        wolves = jnp.sum(state.role * state.status)
        villagers = jnp.sum(state.status) - wolves

        # Check if the wolves have won
        state, rewards = lax.cond(
            (wolves >= villagers),
            lambda _: self._wolf_win(state, rewards),
            lambda _: (state, rewards),
            operand=None,
        )

        # # Check if the villagers have won
        state, rewards = lax.cond(
            (wolves == 0),
            lambda _: self._villager_win(state, rewards),
            lambda _: (state, rewards),
            operand=None,
        )

        # # Check if the time has elapsed
        state, rewards = lax.cond(
            (state.day >= self.MAX_DAY - 1),
            lambda _: self._timeout(state, rewards),
            lambda _: (state, rewards),
            operand=None,
        )

        return state, rewards

    def _next_phase(self, state: State) -> State:
        new_phase = (state.phase + 1) % WerewolfGame.NUM_PHASES
        new_day = state.day + (new_phase == 0)
        return state.replace(phase=new_phase, day=new_day)

    def _update_status(
        self, state: State, target: chex.Array, status: chex.Array
    ) -> State:
        """Updates the status of the target."""
        chex.assert_shape([target, status], ())
        return state.replace(status=state.status.at[target].set(status))

    def _update_vote(self, state: State, agent: AgentID, target: chex.Array) -> State:
        """Updates the target of the agent."""
        chex.assert_shape(target, ())
        return state.replace(votes=state.votes.at[agent, target].set(1))

    def _update_target(self, state: State, agent: AgentID, target: chex.Array) -> State:
        """Updates the target of the wolf agent."""
        chex.assert_shape(target, ())
        return state.replace(targets=state.targets.at[agent, target].set(1))

    def _update_self_votes(
        self, state: State, day: int, phase: int, count: chex.Array
    ) -> State:
        chex.assert_shape(count, ())
        return state.replace(self_votes=state.self_votes.at[day, phase].set(count))

    def _update_kill_dead(
        self, state: State, day: int, phase: int, dead: chex.Array
    ) -> State:
        chex.assert_shape(dead, ())
        return state.replace(kill_dead=state.kill_dead.at[day, phase].set(dead))

    def _update_cannibalism(
        self, state: State, day: int, phase: int, wolf: chex.Array
    ) -> State:
        chex.assert_shape(wolf, ())
        return state.replace(cannibalism=state.cannibalism.at[day, phase].set(wolf))

    def _wolf_win(self, state: State, rewards: chex.Array) -> Tuple[State, chex.Array]:
        state = state.replace(finished=True, winner=Winner.WOLVES)

        rewards = rewards + state.role * self._reward_config.victory
        rewards = rewards + (1 - state.role) * self._reward_config.loss

        return state, rewards

    def _villager_win(
        self, state: State, rewards: chex.Array
    ) -> Tuple[State, chex.Array]:
        state = state.replace(finished=True, winner=Winner.VILLAGERS)

        rewards = rewards + (1 - state.role) * self._reward_config.victory
        rewards = rewards + state.role * self._reward_config.loss

        return state, rewards

    def _timeout(self, state: State, rewards: chex.Array) -> Tuple[State, chex.Array]:
        state = state.replace(finished=True, winner=Winner.TIMEOUT)
        rewards = rewards + self._reward_config.loss
        return state, rewards

    # Observation and rendering ================================================
    def get_obs(self, state: State) -> Dict[AgentID, chex.Array]:
        """Applies observation function to state."""

        def _agent_observation(agent_idx: int):
            return jnp.concatenate(
                [
                    jax.nn.one_hot(state.day, WerewolfGame.MAX_DAY),
                    jax.nn.one_hot(state.phase, WerewolfGame.NUM_PHASES),
                    jnp.stack([agent_idx]),
                    state.status,
                    state.role * (state.role[agent_idx]),
                    jnp.ravel(state.votes),
                    jnp.ravel(state.targets * (state.role[agent_idx])),
                ]
            )

        obs = {a: _agent_observation(i) for i, a in enumerate(self.agents)}
        return obs

    def render(self, state: State):
        """Renders the state of the environment."""
        self._renderer.render(
            state.day, state.phase, state.role, state.status, state.votes, state.targets
        )

    # Utility functions=========================================================
    def _no_op(self, key, state, actions, rewards):
        return state, actions, rewards

    def _compute_target(
        self, key: chex.PRNGKey, target_matrix: chex.Array
    ) -> chex.Array:
        chex.assert_shape(target_matrix, (self.num_agents, self.num_agents))

        targets = jnp.sum(target_matrix, axis=0)
        at_risk = (targets == jnp.max(targets)).astype(int)
        return jnp.argmax(at_risk * jr.uniform(key, at_risk.shape))

    # Other JAX-MARL environment interface functions ===========================
    @property
    def name(self) -> str:
        """Environment name."""
        return "Werewolf"

    @property
    def agent_classes(self) -> dict:
        """Agent classes."""
        raise {"werewolf": ["werewolf"], "villager": ["villager"]}

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]


def text_render(state: State):
    x = ["W" if r == 1 else "V" for r in state.role.tolist()]
    y = ["X" if s == 0 else " " for s in state.status.tolist()]
    print("P:", " ".join(x))
    print("D:", " ".join(y))


def example():
    print("Werewolf example")
    num_agents = 8
    key = jr.PRNGKey(0)

    from jaxmarl import make

    env = make("werewolf", num_agents=num_agents, num_wolves=2)

    obs, state = env.reset(key)
    state: State = state

    winners = []
    while len(winners) < 10:
        print("Day:", state.day, "Phase:", Phase.label(state.phase))

        key, key_act, key_step = jax.random.split(key, 3)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions: Actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        # Perform the step transition.
        obs, state, reward, dones, infos = env.step(key_step, state, actions)
        text_render(state)

        if state.finished:
            winners.append(Winner.label(state.winner))
            print("Winner:", Winner.label(state.winner))
            print("")

    print("Winners:", winners)


if __name__ == "__main__":
    # with jax.disable_jit():
    #     example()
    example()
