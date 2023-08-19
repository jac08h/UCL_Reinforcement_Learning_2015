import statistics
from collections import defaultdict
from random import choice, choices
from typing import Tuple, List, Dict
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from easy21.data import Action, State, Policy, Episode
from easy21.game import generate_episode_from_policy, get_all_possible_actions


def monte_carlo(policy: Policy, n_terminate: int, first_visit: bool = False, n_0: float = 100):
    # See page 99 in Sutton & Barto book for pseudocode
    q_values: Dict[State, Dict[Action, float]] = {}  # Q
    counts: Dict[State, Dict[Action, int]] = {}  # N
    returns: Dict[State, Dict[Action, List[float]]] = {}

    possible_actions = get_all_possible_actions()

    for _ in range(n_terminate):
        episode = generate_episode_from_policy(policy)
        G = 0
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G += reward
            if first_visit and same_state_action_exists_before(episode, i):
                continue

            if state not in counts:
                counts[state] = defaultdict(int)
            if state not in returns:
                returns[state] = defaultdict(list)
            if state not in q_values:
                q_values[state] = {}

            counts[state][action] += 1
            returns[state][action].append(G)

            q_values[state][action] = statistics.mean(returns[state][action])
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

    return q_values, policy, returns


def same_state_action_exists_before(history: Episode, i: int) -> bool:
    query_state, query_action, _ = history[i]
    for state, action, _ in history[:i]:
        if query_state == state and query_action == action:
            return True
    return False


def plot_policy(Q):
    player_sums = []
    dealer_cards = []
    values = []
    for state, actions in Q.items():
        player_sums.append(state.player_sum)
        dealer_cards.append(state.dealer_card)
        values.append(max(actions.values()))

    # Convert the lists to numpy arrays for easier manipulation
    x = np.array(dealer_cards)
    y = np.array(player_sums)
    z = np.array(values)

    # Define a grid for interpolation
    # Create a grid of points based on the range of x and y values
    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

    # Interpolate values to create a smooth surface
    # Use the griddata function to interpolate z values onto the defined grid
    z_interp = griddata((x, y), z, (x_grid, y_grid), method='linear')

    # Create a new figure
    fig = plt.figure()

    # Add a 3D subplot to the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface using the interpolated data
    surf = ax.plot_surface(x_grid, y_grid, z_interp, cmap='viridis')

    # Set labels for the axes
    ax.set_xlabel('Dealer card')
    ax.set_ylabel('Player sum')

    ax.set_zticks([])

    # Display the plot
    plt.show()


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


if __name__ == '__main__':
    policy = {}
    Q, policy, returns = monte_carlo(policy, 100000)
    plot_policy(Q)
