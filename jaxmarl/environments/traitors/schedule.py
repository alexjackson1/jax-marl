from typing import TYPE_CHECKING, NamedTuple

from .shared import (
    Config,
    Activity,
    BreakfastPhase,
    ChallengePhase,
    Phase,
    RoundtablePhase,
    SecretMeetingPhase,
    EndgamePhase,
)

if TYPE_CHECKING:
    from .interface import State


class Timestep(NamedTuple):
    """Time step of the game."""

    day: int
    activity: Activity
    phase: Phase
    phase_step: int
    finished: bool


def next_timestep(state: "State", config: Config) -> Timestep:
    if state.activity == Activity.BREAKFAST:
        if state.phase == BreakfastPhase.GROUP_DISCUSSION:
            if state.phase_step < config.open_length:
                return Timestep(
                    state.day,
                    state.activity,
                    state.phase,
                    state.phase_step + 1,
                    state.finished,
                )
            else:
                return Timestep(
                    state.day,
                    Activity.CHALLENGE,
                    ChallengePhase.ATTEMPT_SHIELD,
                    1,
                    state.finished,
                )
        else:
            raise ValueError("Invalid breakfast phase")

    elif state.activity == Activity.CHALLENGE:
        if state.phase == ChallengePhase.ATTEMPT_SHIELD:
            return Timestep(
                state.day,
                Activity.ROUNDTABLE,
                RoundtablePhase.GROUP_DISCUSSION,
                1,
                state.finished,
            )
        else:
            raise ValueError("Invalid challenge phase")

    elif state.activity == Activity.ROUNDTABLE:
        if state.phase == RoundtablePhase.GROUP_DISCUSSION:
            if state.phase_step < config.open_length:
                return Timestep(
                    state.day,
                    state.activity,
                    state.phase,
                    state.phase_step + 1,
                    state.finished,
                )
            else:
                return Timestep(
                    state.day,
                    Activity.ROUNDTABLE,
                    RoundtablePhase.ROUNDTABLE_VOTE,
                    1,
                    state.finished,
                )

        elif state.phase == RoundtablePhase.ROUNDTABLE_VOTE:
            if state.phase_step == 1:
                if state.banishment_selection is None:
                    return Timestep(
                        state.day,
                        Activity.ROUNDTABLE,
                        RoundtablePhase.ROUNDTABLE_VOTE,
                        2,
                        state.finished,
                    )
                else:
                    return Timestep(
                        state.day,
                        Activity.SECRET_MEETING,
                        SecretMeetingPhase.TRAITORS_DISCUSSION,
                        1,
                        state.finished,
                    )
            elif state.phase_step == 2:
                return Timestep(
                    state.day,
                    Activity.SECRET_MEETING,
                    SecretMeetingPhase.TRAITORS_DISCUSSION,
                    1,
                    state.finished,
                )
            else:
                raise ValueError("Invalid roundtable phase step")

    elif state.activity == Activity.SECRET_MEETING:
        if state.phase == SecretMeetingPhase.TRAITORS_DISCUSSION:
            if state.phase_step < config.hidden_length:
                return Timestep(
                    state.day,
                    state.activity,
                    state.phase,
                    state.phase_step + 1,
                    state.finished,
                )
            else:
                return Timestep(
                    state.day,
                    Activity.SECRET_MEETING,
                    SecretMeetingPhase.TRAITORS_ACTION_VOTE,
                    1,
                    state.finished,
                )

        elif state.phase == SecretMeetingPhase.TRAITORS_ACTION_VOTE:
            if state.phase_step == 1:
                if state.traitor_action_selection is None:
                    return Timestep(
                        state.day,
                        state.activity,
                        state.phase,
                        state.phase_step + 1,
                        state.finished,
                    )
                else:
                    return Timestep(
                        state.day,
                        Activity.SECRET_MEETING,
                        SecretMeetingPhase.TRAITORS_TARGET_VOTE,
                        1,
                        state.finished,
                    )
            elif state.phase_step == 2:
                return Timestep(
                    state.day,
                    Activity.SECRET_MEETING,
                    SecretMeetingPhase.TRAITORS_TARGET_VOTE,
                    1,
                    state.finished,
                )
            else:
                raise ValueError("Invalid secret meeting phase step")

        elif state.phase == SecretMeetingPhase.TRAITORS_TARGET_VOTE:
            if state.phase_step == 1:
                return Timestep(
                    state.day,
                    state.activity,
                    state.phase,
                    state.phase_step + 1,
                    state.finished,
                )
            elif state.phase_step == 2:
                return Timestep(
                    state.day,
                    Activity.SECRET_MEETING,
                    SecretMeetingPhase.RECRUITMENT_OFFER,
                    1,
                    state.finished,
                )
            else:
                raise ValueError("Invalid secret meeting phase step")

        elif state.phase == SecretMeetingPhase.RECRUITMENT_OFFER:
            if state.day == config.num_days - 1:
                return Timestep(
                    state.day,
                    Activity.ENDGAME,
                    EndgamePhase.ENDGAME_VOTE,
                    1,
                    state.finished,
                )
            else:
                return Timestep(
                    state.day + 1,
                    Activity.BREAKFAST,
                    BreakfastPhase.GROUP_DISCUSSION,
                    1,
                    state.finished,
                )

        else:
            raise ValueError("Invalid secret meeting phase")

    elif state.activity == Activity.ENDGAME:
        if state.phase == EndgamePhase.ENDGAME_VOTE:
            if state.banish_again:
                return Timestep(
                    state.day,
                    Activity.ENDGAME,
                    EndgamePhase.ENDGAME_BANISHMENT,
                    1,
                    state.finished,
                )
            else:
                return Timestep(
                    state.day,
                    Activity.ENDGAME,
                    EndgamePhase.ENDGAME_BANISHMENT,
                    2,
                    True,
                )

        elif state.phase == EndgamePhase.ENDGAME_BANISHMENT:
            if state.phase_step == 1:
                if config.num_agents - sum(state.murdered + state.banished) <= 2:
                    return Timestep(
                        state.day,
                        state.activity,
                        state.phase,
                        state.phase_step + 1,
                        state.finished,
                    )
                if state.banishment_selection is None:
                    return Timestep(
                        state.day,
                        state.activity,
                        state.phase,
                        state.phase_step + 1,
                        state.finished,
                    )
                else:
                    return Timestep(
                        state.day,
                        Activity.ENDGAME,
                        EndgamePhase.ENDGAME_VOTE,
                        1,
                        state.finished,
                    )
            elif state.phase_step == 2:
                return Timestep(
                    state.day,
                    Activity.ENDGAME,
                    EndgamePhase.ENDGAME_VOTE,
                    1,
                    state.finished,
                )

        else:
            raise ValueError("Invalid endgame phase")

    else:
        raise ValueError("Invalid activity")

    return Timestep(step=state.step + 1)
