from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import chex

if TYPE_CHECKING:
    from .logic import State


class TextRenderer:
    RED = "\033[91m"
    GREEN = "\033[92m"
    ORANGE = "\033[93m"
    PURPLE = "\033[95m"
    GREY = "\033[90m"
    BOLD = "\033[1m"
    STRIKE = "\033[9m"
    END = "\033[0m"

    @classmethod
    def state(cls, state: "State"):
        f = lambda r, s: cls.wolf(s == 0) if r == 1 else cls.villager(s == 0)
        return " ".join([f(r, s) for r, s in zip(state.role, state.status)])

    @classmethod
    def wolf(cls, dead: bool):
        x = cls.BOLD + cls.PURPLE if not dead else cls.GREY
        return f"{x}W{cls.END}"

    @classmethod
    def villager(cls, dead: bool):
        x = cls.BOLD + cls.ORANGE if not dead else cls.GREY
        return f"{x}V{cls.END}"


class PlotRenderer:

    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    @property
    def block_size(self) -> float:
        """Size of the blocks."""
        Y_1, Y_2 = 0.05, 0.02
        X_1, X_2 = 4, 15
        M = (Y_1 - Y_2) / (X_1 - X_2)
        C = Y_2 - M * X_2
        return M * self.num_agents + C

    @property
    def block_spacing(self) -> float:
        """Size of the blocks."""
        Y_1, Y_2 = 0.025, 0.01
        X_1, X_2 = 4, 15
        M = (Y_1 - Y_2) / (X_1 - X_2)
        C = Y_2 - M * X_2
        return M * self.num_agents + C

    @property
    def title_height(self) -> float:
        """Size of the title."""
        return 0.05

    @property
    def margin(self) -> float:
        """Size of the margin."""
        return 0.05

    @property
    def votes_height(self) -> float:
        """Size of the bottom panel."""
        return (
            self.num_agents * (self.block_size + self.block_spacing)
            + self.block_spacing
            + self.title_height
            + self.margin
        )

    @property
    def votes_width(self) -> float:
        """Size of the bottom panel."""
        return (
            self.num_agents * (self.block_size + self.block_spacing)
            + self.block_spacing
            + self.margin
        )

    @property
    def players_height(self) -> float:
        """Size of the middle panel."""
        return (
            self.block_size + self.block_spacing * 2 + self.title_height + self.margin
        )

    def render(
        self,
        day: int,
        phase: int,
        role: chex.Array,
        status: chex.Array,
        targets: chex.Array,
        signals: chex.Array,
    ):
        """Renders the state of the environment."""
        fig, ax = plt.subplots(figsize=(6, 6))

        t = ax.transAxes
        self._plt_time(ax, day, phase, t)
        self._plt_players(ax, role, status, t)
        self._plt_votes(ax, targets, t)
        self._plt_comm_matrix(ax, signals, t)

        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def _plt_time(self, ax: plt.Axes, day: int, phase: int, transform=None):
        """Plots time."""
        offset = (0, self.votes_height + self.players_height)
        time_text = f"DAY {day} (PHASE {phase})"
        plt.text(
            offset[0],
            offset[1],
            time_text,
            fontsize=14,
            fontweight="bold",
            ha="left",
            transform=transform,
        )

    def _plt_players(
        self, ax: plt.Axes, role: chex.Array, status: chex.Array, transform=None
    ):
        offset = (0, self.votes_height)
        size, spacing = self.block_size, self.block_spacing

        plt.text(
            offset[0],
            offset[1] + size + spacing * 2,
            "PLAYERS",
            transform=transform,
            fontsize=12,
            fontweight="bold",
            ha="left",
        )

        for i, player in enumerate(role):
            color = "seagreen" if player == 0 else "slategray"
            color = "black" if status[i] == 0 else color

            ax.add_patch(
                plt.Rectangle(
                    xy=(offset[0] + (size + spacing) * i, offset[1]),
                    width=size,
                    height=size,
                    color=color,
                    transform=transform,
                ),
            )

    def _plt_votes(self, ax: plt.Axes, targets: chex.Array, transform=None):
        """Plots targets."""
        offset = (0, 0)
        size, spacing = self.block_size, self.block_spacing

        # plot title
        plt.text(
            offset[0],
            offset[1] + (((size + spacing) * self.num_agents) + spacing),
            "VOTES",
            transform=transform,
            fontsize=12,
            fontweight="bold",
            ha="left",
        )
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                ax.add_patch(
                    plt.Rectangle(
                        xy=(
                            offset[0] + (size + spacing) * i,
                            offset[1] + (size + spacing) * j,
                        ),
                        width=size,
                        height=size,
                        color="black",
                        fill=True,
                        transform=transform,
                    )
                )

    def _plt_comm_matrix(self, ax: plt.Axes, comm_matrix: chex.Array, transform=None):
        """Plots targets."""
        offset = (self.votes_width, 0)
        size, spacing = self.block_size, self.block_spacing

        # plot title
        plt.text(
            offset[0],
            offset[1] + (((size + spacing) * self.num_agents) + spacing),
            "SIGNALS",
            transform=transform,
            fontsize=12,
            fontweight="bold",
            ha="left",
        )
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                ax.add_patch(
                    plt.Rectangle(
                        xy=(
                            offset[0] + (size + spacing) * i,
                            offset[1] + (size + spacing) * j,
                        ),
                        width=size,
                        height=size,
                        color="black",
                        fill=True,
                        transform=transform,
                    )
                )
