from easy21.data import Policy, Episode
from easy21.game import generate_episode_from_policy, initialize_memory
from easy21.graphics import plot_policy
from easy21.utils import epsilon_greedy_policy_improvement


def monte_carlo(policy: Policy,
                n_terminate: int,
                first_visit: bool = False,
                n_0: float = 100):
    q_values = initialize_memory(0.0)  # Q(s, a)
    counts = initialize_memory(0.0)  # N(s, a)

    for _ in range(n_terminate):
        episode = generate_episode_from_policy(policy)
        G = 0
        for i in range(len(episode)):
            state, action, reward = episode[i]
            G += reward
            if first_visit and same_state_action_exists_before(episode, i):
                continue

            counts[state][action] += 1
            q_values[state][action] += (1 / counts[state][action]) * (G - q_values[state].get(action, 0))

            eps = n_0 / (n_0 + sum(action_count for action_count in counts[state].values()))
            epsilon_greedy_policy_improvement(policy, state, q_values, eps)

    return q_values, policy


def same_state_action_exists_before(history: Episode, i: int) -> bool:
    query_state, query_action, _ = history[i]
    for state, action, _ in history[:i]:
        if query_state == state and query_action == action:
            return True
    return False


if __name__ == '__main__':
    policy = {}
    Q, policy = monte_carlo(policy, 10000)
    plot_policy(Q)
