from typing import Dict

from easy21.data import Policy, State, Action
from easy21.game import generate_episode_from_policy, ALL_POSSIBLE_ACTIONS


def simulate_games(policy, n):
    wins = loses = draws = 0
    for _ in range(n):
        episode = generate_episode_from_policy(policy)
        if episode[-1][2] == 1:
            wins += 1
        elif episode[-1][2] == -1:
            loses += 1
        else:
            draws += 1
    return wins, draws, loses


def epsilon_greedy_policy_improvement(policy: Policy,
                                      state: State,
                                      q_values: Dict[State, Dict[Action, float]],
                                      eps: float
                                      ) -> None:
    best_action = best_action_value = None  # A*
    for action, value in q_values[state].items():
        if best_action is None or value > best_action_value:
            best_action = action
            best_action_value = value

    for action in ALL_POSSIBLE_ACTIONS:
        if state not in policy:
            policy[state] = {}
        if action == best_action:
            policy[state][action] = 1 - eps + eps / len(ALL_POSSIBLE_ACTIONS)
        else:
            policy[state][action] = eps / len(ALL_POSSIBLE_ACTIONS)