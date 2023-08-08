from dataclasses import dataclass
from random import randint, random
from enum import Enum, auto
from typing import Optional, Tuple


class Action(Enum):
    HIT = auto()
    STICK = auto()


@dataclass
class State:
    dealer_card: int
    player_sum: int


def new_card() -> int:
    value = randint(1, 11)
    if random() < 1 / 3:  # red card
        value = -value
    return value


def is_busted(current_sum: int) -> bool:
    return current_sum < -1 or current_sum > 21


def initialize_state() -> State:
    return State(
        abs(new_card()),
        abs(new_card())
    )


def step(state: Optional[State], action: Action) -> Tuple[Optional[State], int]:
    if state is None:
        raise ValueError("Calling step with terminal state")

    if action == Action.HIT:
        state.player_sum += new_card()
        if is_busted(state.player_sum):
            return None, -1
        return state, 0

    elif action == Action.STICK:
        dealer_sum = state.dealer_card
        while dealer_sum < 17:
            dealer_sum += new_card()
            if is_busted(dealer_sum):
                return None, 1

        # compare
        if state.player_sum > dealer_sum:
            return None, 1
        elif state.player_sum == dealer_sum:
            return None, 0
        else:
            return None, -1
