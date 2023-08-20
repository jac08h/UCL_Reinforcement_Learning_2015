from collections import defaultdict
from typing import Dict, Tuple

from easy21.data import Action, State, Policy, Q_values
from easy21.game import sample_policy, initialize_state, step, ALL_POSSIBLE_ACTIONS, initialize_memory, \
    all_possible_states
from easy21.utils import epsilon_greedy_policy_improvement


def sarsa_lambda(policy: Policy,
                 lmbd: float,
                 n_terminate: int,
                 n_0: float = 100
                 ) -> Tuple[Policy, Q_values]:
    q_values = initialize_memory(0.0)  # Q(s, a)
    q_values[None] = defaultdict(int)
    counts = initialize_memory(0.0)  # N(s, a)
    discount_factor = 1  # gamma

    for _ in range(n_terminate):
        eligibility_traces: Dict[State, Dict[Action, int]] = initialize_memory(0.0)
        state = initialize_state()
        while state is not None:
            # Take action A, observe R, S'
            action = sample_policy(policy, state)
            new_state, reward = step(state, action)
            # Choose A' from S' using policy derived from Q
            new_action = sample_policy(policy, new_state)

            delta = reward + (discount_factor * q_values[new_state].get(new_action, 0)) - q_values[state].get(action, 0)
            eligibility_traces[state][action] += 1
            counts[state][action] += 1
            learning_rate = (1 / counts[state][action])  # alpha
            for s in all_possible_states():
                for a in ALL_POSSIBLE_ACTIONS:
                    q_values[s][a] += learning_rate * delta * eligibility_traces[s][a]
                    eligibility_traces[s][a] *= discount_factor * lmbd

            eps = n_0 / (n_0 + sum(action_count for action_count in counts[state].values()))
            epsilon_greedy_policy_improvement(policy, state, q_values, eps)

            state = new_state

    return q_values, policy


if __name__ == '__main__':
    q_values, policy = sarsa_lambda({}, 1, 1000)
