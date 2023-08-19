from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Tuple


class Action(Enum):
    HIT = auto()
    STICK = auto()


@dataclass
class State:
    dealer_card: int
    player_sum: int

    def __hash__(self):
        return hash(self.dealer_card ** 2 * self.player_sum ** 3)


Policy = Dict[State, Dict[Action, float]]
Episode = List[Tuple[State, Action, float]]
