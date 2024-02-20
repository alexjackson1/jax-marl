""" 
A social deduction game based on the BBC television series 'The Traitors'.

The players of the game are secretly divided into two teams: the traitors and 
the faithful.
All players are incentivised to make it to the end of the game; however, the 
faithful can only win if they identify and eliminate all the traitors, while
the traitors must only make it to the end of the game without being eliminated.
The game proceeds in rounds, with each round consisting of the following
activities.

1. Breakfast.
    All players are made aware of the identity of any players murdered 
    during the night, and are given the opportunity to discuss the game. As 
    traitors are only able to murder faithful, any murdered players were
    necessarily faithful.

2. Challenge.
    The players are given a chance to attempt a fixed number of shields that
    protects them from being murdered during the night.
    The chance of obtaining a shield once attempted is determined randomly.

3. Roundtable.
    The players discuss and vote on who to eliminate from the game.
    In the event of a tie, the players with the most votes are then voted on 
    again by the remaining players.
    If the outcome is still a tie, then one player from the tied players is
    randomly selected to be eliminated. 
    Once a player is eliminated, their identity is revealed to all players.

4. Secret Meeting.
    The traitors meet secretly to discuss and vote on one of two actions, if
    there is no consensus a random action is immediately chosen.
    - Every night the traitors have the option to eliminate one faithful.
      The faithful to be eliminated is selected in the same manner as the
      roundtable elimination vote.
    - If there are less than three traitors remaining in the game, the traitors 
      are given the opportunity to recruit a new member from the faithful.
      The new member is selected in the same manner as the roundtable
      elimination vote; however, the new member must either accept or decline
      the offer to join the traitors.
      If the new member accepts, they are immediately informed of the identity
      of the other traitors and the game continues.
      If the new member declines, the game continues as normal.

"""

from .interface import TraitorsGame, State
