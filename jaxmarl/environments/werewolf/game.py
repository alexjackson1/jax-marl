from typing import Dict, List, Literal, Tuple

from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr

import chex

from .logic import (
    WerewolfLogic,
    GameConfig,
    RewardConfig,
    State,
    GameStatus,
    AgentID,
    Actions,
)
from .render import TextRenderer

from ..multi_agent_env import MultiAgentEnv
from ..spaces import Discrete as DiscreteSpace, Dict as DictSpace


class WerewolfGame(MultiAgentEnv):
    """A mixed-motive multi-agent communication implementation of Werewolf."""

    config: GameConfig
    rewards: RewardConfig
    engine: WerewolfLogic

    num_agents: int
    num_wolves: int
    num_villagers: int

    agents: List[AgentID]
    observation_spaces: Dict[AgentID, chex.Array]
    action_spaces: Dict[AgentID, DictSpace]

    # Initialisation =====================================================================
    def __init__(self, config: GameConfig, rewards: RewardConfig):
        self.config = config
        self.rewards = rewards

        self.num_agents = self.config.num_agents
        self.num_wolves = self.config.num_wolves
        self.num_villagers = self.num_agents - self.num_wolves

        self.villagers, self.wolves = self.init_agents()
        self.agents = self.villagers + self.wolves

        self.observation_spaces = self.init_observation_spaces()
        self.action_spaces = self.init_action_spaces()

        self.engine = WerewolfLogic(config, rewards, self.agents)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Werewolf"

    def init_agents(self) -> Tuple[List[AgentID], List[AgentID]]:
        """Initialises agents."""
        return (
            [f"villager_{i}" for i in range(self.num_villagers)],
            [f"werewolf_{i}" for i in range(self.num_wolves)],
        )

    @property
    def agent_classes(self) -> Dict[Literal["villager", "werewolf"], List[AgentID]]:
        """Agent classes."""
        return {"villager": self.villagers, "werewolf": self.wolves}

    def init_observation_spaces(self) -> Dict[AgentID, chex.Array]:
        """Initialises observation spaces."""
        obs_size = sum(
            [
                self.config.max_day + self.config.num_phases,  # one hot time
                1 + len(self.agents) * 4,  # own id (1), status, role, votes, targets
            ]
        )
        return {id: DiscreteSpace(obs_size) for id in self.agents}

    def init_action_spaces(self) -> Dict[AgentID, DictSpace]:
        """Initialises action spaces."""
        return {
            id: DictSpace({"target": DiscreteSpace(len(self.agents))})
            for id in self.agents
        }

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    # Reset ==============================================================================
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[AgentID, chex.Array], State]:
        """Performs resetting of the environment."""
        id_shuff, role_shuff = jr.split(key, 2)
        counts = (self.num_wolves, self.num_villagers)
        roles = WerewolfGame.assign_roles(role_shuff, counts)

        ids = jr.permutation(id_shuff, self.num_agents)
        state = State.create(ids, roles, self.config.max_day)
        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state)

    @staticmethod
    def assign_roles(key: chex.PRNGKey, counts: Tuple[int, int]) -> chex.Array:
        """Assigns roles to agents."""
        wc, vc = counts
        roles = jnp.concatenate([jnp.full((wc), 1), jnp.full((vc), 0)])
        return jr.permutation(key, roles)

    # Step and transition functions ======================================================
    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Actions
    ) -> Tuple[Dict[str, chex.Array], State, chex.Array, Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        # Initialise the reward array and info dict
        rewards, infos = jnp.zeros(self.num_agents), {}

        def f1(key, state, actions, rewards):
            # Update the state w.r.t. the actions
            state, actions, rewards = self.engine.pre_actions(state, actions, rewards)
            state, actions, rewards = self.engine.run_actions(
                key, state, actions, rewards
            )
            state = self.engine.step_time(state)
            return state, actions, rewards, False

        def f2(state, actions, rewards):
            return state, actions, rewards, True

        state, actions, rewards, is_done = lax.cond(
            state.finished,
            lambda _: f2(state, actions, rewards),
            lambda _: f1(key, state, actions, rewards),
            operand=None,
        )

        state, rewards = self.engine.post_actions(state, rewards)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            lax.stop_gradient(rewards),
            {"__all__": is_done},
            infos,
        )

    # Observation and rendering ================================================
    def get_obs(self, state: State) -> Dict[AgentID, chex.Array]:
        """Applies observation function to state."""

        def _agent_observation(agent_idx: int):
            return jnp.concatenate(
                [
                    jax.nn.one_hot(state.day, self.config.max_day),
                    jax.nn.one_hot(state.phase, self.config.num_phases),
                    jnp.stack([agent_idx]),
                    state.status,
                    state.role * (state.role[agent_idx]),
                    jnp.ravel(state.votes),
                    jnp.ravel(state.targets * (state.role[agent_idx])),
                ]
            )

        obs = {id: _agent_observation(i) for i, id in enumerate(self.agents)}
        return obs

    def render(self, state: State):
        """Renders the state of the environment."""
        print(TextRenderer.state(state), end="\r")


def example():
    print("Werewolf example")
    key = jr.PRNGKey(0)

    from jaxmarl import make

    env = make(
        "werewolf",
        config=GameConfig(num_agents=8, num_wolves=2, max_day=10),
        rewards=RewardConfig(),
    )

    obs, state = env.reset(key)
    state: State = state

    winners = []
    while len(winners) < 10:
        key, key_act, key_step = jax.random.split(key, 3)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions: Actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        # Perform the step transition.
        obs, state, reward, dones, infos = env.step(key_step, state, actions)
        env.render(state)

        if state.finished:
            winners.append(GameStatus.label(state.game_status))


if __name__ == "__main__":
    # with jax.disable_jit():
    #     example()
    example()
