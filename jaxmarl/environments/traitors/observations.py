from ..spaces import Dict as DictSpace, Discrete, MultiDiscrete


class ObservationSpace(DictSpace):
    def __init__(
        self,
        num_agents: int,
        num_days: int,
        num_activites: int,
        max_phases: int,
        max_phase_steps: int,
        num_open_symbols: int,
        num_hidden_symbols: int,
    ):
        super().__init__(
            {
                "player_id": Discrete(num_agents),
                "timestep": DictSpace(
                    {
                        "day": Discrete(num_days),
                        "activity": Discrete(num_activites),
                        "phase": Discrete(max_phases),
                        "phase_step": Discrete(max_phase_steps),
                    }
                ),
                "player_stats": DictSpace(
                    {
                        "is_dead": Discrete(num_agents),
                        "is_traitor": Discrete(num_agents),
                        "attempted_shield": Discrete(num_agents),
                        "has_shield": Discrete(num_agents),
                        "at_risk": Discrete(num_agents),
                    }
                ),
                "votes": DictSpace(
                    {
                        "roundtable": MultiDiscrete([num_agents, num_agents]),
                        "traitor_action": MultiDiscrete([num_agents, 2]),
                        "traitor_target": MultiDiscrete([num_agents, num_agents]),
                        "endgame": MultiDiscrete([num_agents, 2]),
                    }
                ),
                "communication": DictSpace(
                    {
                        "open": MultiDiscrete(
                            [num_agents, num_open_symbols, num_agents]
                        ),
                        "hidden": MultiDiscrete(
                            [num_agents, num_hidden_symbols, num_agents]
                        ),
                    }
                ),
            }
        )
