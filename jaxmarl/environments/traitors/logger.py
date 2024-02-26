import sys

import logging
import textwrap

import jax


class GameLogger(logging.Logger):
    """A custom logger for the game."""

    MESSAGE_PREFIX = "|-"
    SPACE_PREFIX = "|"
    TITLE_PREFIX = "|==="
    EVENT_PREFIX = "|---"

    H1_CHAR = "="
    H2_CHAR = "-"

    ITALIC = "\033[3m"
    BOLD = "\033[1m"
    COLOR_ORANGE = "\033[33m"
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_BLUE = "\033[94m"
    RESET_FORMATTING = "\033[0m"

    def __init__(self, name, width: int = 79, color: bool = True):
        """
        Initialise a GameLogger object.

        Args:
            name (str): The name of the logger.
            width (int): The width of the log messages.
            color (bool): Whether to use ANSI color codes in the output.
        """
        super().__init__(name)
        self.width = width

        # check if jax_disable_jit is set
        if jax.config.jax_disable_jit:
            formatter = logging.Formatter("%(message)s")
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            self.addHandler(stdout_handler)

        if not color:
            self.ITALIC = ""
            self.BOLD = ""
            self.COLOR_ORANGE = ""
            self.COLOR_RED = ""
            self.COLOR_GREEN = ""
            self.COLOR_BLUE = ""
            self.RESET_FORMATTING = ""

    def set_debug(self, debug: bool):
        """
        Set the debug level of the logger.

        Args:
            debug (bool): Whether to enable debug logging.
        """
        if debug:
            self.setLevel(logging.DEBUG)
        else:
            self.setLevel(logging.INFO)

    def title(self):
        super().info(self._ascii_art())
        super().info(self._app_info())

    def debug(self, msg, *args, **kwargs):
        """
        Log a message with severity DEBUG.

        Args:
            msg (str): The message to be logged.
            *args: Additional arguments to be passed to the logger.
            **kwargs: Additional keyword arguments to be passed to the logger.
        """
        init, post = self.MESSAGE_PREFIX, self.SPACE_PREFIX
        post += (len(init) - len(post)) * " "
        lines = textwrap.fill(msg, self.width - len(init)).split("\n")

        s = ""
        for i, wrapped_line in enumerate(lines):
            italic_text = self._fmt_italic(wrapped_line)
            s += f"{init} {italic_text}" if i == 0 else f"{post} {italic_text}"

            if i < len(lines) - 1:
                s += "\n"

        super().debug(s, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        init, post = self.MESSAGE_PREFIX, self.SPACE_PREFIX
        post += (len(init) - len(post)) * " "
        lines = textwrap.fill(msg, self.width - len(init)).split("\n")

        s = ""
        for i, wrapped_line in enumerate(lines):
            bold_prefix = self._fmt_bold(init) if i == 0 else self._fmt_bold(post)
            s += f"{bold_prefix} {wrapped_line}"

            if i < len(lines) - 1:
                s += "\n"

        super().info(s, *args, **kwargs)

    def heading_1(self, day: int, *args, **kwargs):
        prefix = f"{self.TITLE_PREFIX} DAY {day} "
        assert len(prefix) <= self.width

        heading = f"{prefix}{self.H1_CHAR * (self.width - len(prefix))}"
        super().info(self._fmt_bold(f"{self.COLOR_ORANGE}{heading}"), *args, **kwargs)

    def heading_2(self, label: str, *args, **kwargs):
        prefix = f"{self.TITLE_PREFIX} {label.upper()} "
        assert len(prefix) <= self.width

        heading = f"{prefix}{self.H2_CHAR * (self.width - len(prefix))}"
        super().info(self._fmt_bold(heading), *args, **kwargs)

    def blank(self):
        super().info(self._fmt_bold(self.SPACE_PREFIX))

    def _fmt_italic(self, msg: str) -> str:
        return f"{self.ITALIC}{msg}{self.RESET_FORMATTING}"

    def _fmt_bold(self, msg: str) -> str:
        return f"{self.BOLD}{msg}{self.RESET_FORMATTING}"

    def _ascii_art(self):
        return f"""{self.BOLD}{self.COLOR_ORANGE}
888888 88  88 888888     888888 88""Yb    db    88 888888  dP"Yb  88""Yb .dP"Y8 
  88   88  88 88__         88   88__dP   dPYb   88   88   dP   Yb 88__dP `Ybo." 
  88   888888 88""         88   88"Yb   dP__Yb  88   88   Yb   dP 88"Yb  o.`Y8b 
  88   88  88 888888       88   88  Yb dP''''Yb 88   88    YbodP  88  Yb 8bodP'
    {self.RESET_FORMATTING}"""

    def _app_info(self):
        return f"""{self.BOLD}An agent-based simulation model of the BBC TV show "The Traitors".{self.RESET_FORMATTING}
{self.ITALIC}This is a research project and unaffiliated with the BBC or Studio Lambert.{self.RESET_FORMATTING}

Author: Alex Jackson <mail@alexjackson.uk>
Credit: BBC/Studio Lambert
License: MIT License
{self.RESET_FORMATTING}"""
