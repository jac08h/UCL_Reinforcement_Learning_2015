from copy import deepcopy
from random import choice, choices
from random import randint, random
from typing import Optional
from typing import Tuple, List

from easy21.data import Policy
from easy21.data import State, Action


def get_all_possible_actions() -> List[Action]:
    return [Action.HIT, Action.STICK]


def new_card() -> int:
    value = randint(1, 11)
    if random() < 1 / 3:  # red card
        value = -value
    return value


def is_busted(current_sum: int) -> bool:
    return current_sum < 1 or current_sum > 21


def initialize_state() -> State:
    return State(
        abs(new_card()),
        abs(new_card())
    )


def step(state: Optional[State], action: Action) -> Tuple[Optional[State], int]:
    assert state is not None

    if action == Action.HIT:
        # TODO necessary?
        new_state = deepcopy(state)
        new_state.player_sum += new_card()
        if is_busted(new_state.player_sum):
            return None, -1
        return new_state, 0

    elif action == Action.STICK:
        dealer_sum = state.dealer_card
        while dealer_sum < 17:
            dealer_sum += new_card()
            if is_busted(dealer_sum):
                return None, 1

        if state.player_sum > dealer_sum:
            return None, 1
        elif state.player_sum == dealer_sum:
            return None, 0
        else:
            return None, -1


def sample_policy(policy: Policy, state: State) -> Action:
    if state in policy:
        actions = list(policy[state].keys())
        probabilities = list(policy[state].values())
        return choices(actions, probabilities, k=1)[0]
    return choice(get_all_possible_actions())


def generate_episode_from_policy(policy: Policy) -> List[Tuple[State, Action, int]]:
    state = initialize_state()
    history = []
    while True:
        action = sample_policy(policy, state)
        new_state, reward = step(state, action)
        assert new_state is None or new_state.player_sum != state.player_sum
        history.append((state, action, reward))
        if new_state is None:
            break
        state = new_state

    return history
