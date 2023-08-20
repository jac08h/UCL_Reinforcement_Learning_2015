from easy21.game import generate_episode_from_policy


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
