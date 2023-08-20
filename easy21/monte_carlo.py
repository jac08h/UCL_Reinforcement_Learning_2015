from collections import defaultdict
from typing import Dict

from easy21.data import Action, State, Policy, Episode
from easy21.game import generate_episode_from_policy, get_all_possible_actions
from easy21.graphics import plot_policy


def monte_carlo(policy: Policy,
                n_terminate: int,
                first_visit: bool = False,
                n_0: float = 100):
    q_values: Dict[State, Dict[Action, float]] = {}  # Q(s, a)
    counts: Dict[State, Dict[Action, int]] = {}  # N(s, a)

    possible_actions = get_all_possible_actions()

    for _ in range(n_terminate):
        episode = generate_episode_from_policy(policy)
        G = 0
        for i in range(len(episode)):
            state, action, reward = episode[i]
            G += reward
            if first_visit and same_state_action_exists_before(episode, i):
                continue

            if state not in counts:
                counts[state] = defaultdict(int)
            if state not in q_values:
                q_values[state] = defaultdict(int)

            counts[state][action] += 1
            # incremental mean
            q_values[state][action] += (1 / counts[state][action]) * (G - q_values[state].get(action, 0))

            eps = n_0 / (n_0 + sum(action_count for action_count in counts[state].values()))
            best_action = best_action_value = None  # A*
            for action, value in q_values[state].items():
                if best_action is None or value > best_action_value:
                    best_action = action
                    best_action_value = value

            for action in possible_actions:
                if state not in policy:
                    policy[state] = {}
                if action == best_action:
                    policy[state][action] = 1 - eps + eps / len(possible_actions)
                else:
                    policy[state][action] = eps / len(possible_actions)

    return q_values, policy


def same_state_action_exists_before(history: Episode, i: int) -> bool:
    query_state, query_action, _ = history[i]
    for state, action, _ in history[:i]:
        if query_state == state and query_action == action:
            return True
    return False


if __name__ == '__main__':
    policy = {}
    Q, policy = monte_carlo(policy, 100000)
    plot_policy(Q)
