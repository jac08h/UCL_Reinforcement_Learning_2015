import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def plot_policy(Q):
    player_sums = []
    dealer_cards = []
    values = []
    for state, actions in Q.items():
        if state is None:
            continue
        player_sums.append(state.player_sum)
        dealer_cards.append(state.dealer_card)
        values.append(max(actions.values()))

    x = np.array(dealer_cards)
    y = np.array(player_sums)
    z = np.array(values)

    x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))

    z_interp = griddata((x, y), z, (x_grid, y_grid), method='linear')

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z_interp, cmap='viridis')

    ax.set_xlabel('Dealer card')
    ax.set_ylabel('Player sum')
    ax.set_zticks([])

    plt.show()
