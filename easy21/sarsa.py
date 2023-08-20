from collections import defaultdict
from typing import Dict

from easy21.data import Action, State, Policy, Episode
from easy21.game import get_all_possible_actions, sample_policy, initialize_state, step
from easy21.graphics import plot_policy


def sarsa(policy: Policy,
          n_terminate: int,
          n_0: float = 100):
    q_values: Dict[State, Dict[Action, float]] = {}  # Q(s, a)
    counts: Dict[State, Dict[Action, int]] = {}  # N(s, a)

    for _ in range(n_terminate):
        state = initialize_state()
        while state is not None:
            if state not in counts:
                counts[state] = defaultdict(int)
            if state not in q_values:
                q_values[state] = defaultdict(int)

            action = sample_policy(policy, state)
            new_state, reward = step(state, action)
            new_action = sample_policy(policy, new_state)

            if new_state not in q_values:
                q_values[new_state] = defaultdict(int)

            counts[state][action] += 1
            step_size = (1 / counts[state][action])  # alpha
            td_target = reward + q_values[new_state].get(new_action, 0)
            q_values[state][action] += step_size * (td_target - q_values[state].get(action, 0))

            eps = n_0 / (n_0 + sum(action_count for action_count in counts[state].values()))
            epsilon_greed_policy_improvement(policy, state, q_values, eps)

            state = new_state

    return q_values, policy


def epsilon_greed_policy_improvement(policy: Policy,
                                     state: State,
                                     q_values: Dict[State, Dict[Action, float]],
                                     eps: float
                                     ) -> None:
    possible_actions = get_all_possible_actions()
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


def same_state_action_exists_before(history: Episode, i: int) -> bool:
    query_state, query_action, _ = history[i]
    for state, action, _ in history[:i]:
        if query_state == state and query_action == action:
            return True
    return False


if __name__ == '__main__':
    policy = {}
    Q, policy = sarsa(policy, 10000)
    plot_policy(Q)
