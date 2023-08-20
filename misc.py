from matplotlib import pyplot as plt

from easy21.graphics import plot_policy
from easy21.monte_carlo import monte_carlo
from easy21.sarsa_lambda import sarsa_lambda
from easy21.utils import compare_q_values

lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
errors = [96.78902301555983, 100.8919161805118, 126.08883421605775, 110.26557203262037, 105.4681207396348,
          106.83095415751018, 122.95285385870953, 140.4796209268404, 186.28750657356554, 188.9751399151779,
          250.5131974762682]


def compare_monte_carlo_to_lambdas():
    monte_carlo_q_values, monte_carlo_policy = monte_carlo({}, 100000)
    plot_policy(monte_carlo_q_values)
    lambdas = []
    errors = []
    for lmbd in range(0, 11):
        lmbd /= 10
        q_values, policy = sarsa_lambda({}, lmbd, 1000)
        error = compare_q_values(monte_carlo_q_values, q_values)
        lambdas.append(lmbd)
        errors.append(error)

    plt.scatter(lambdas, errors, color='blue', marker='o', label='Errors')

    plt.xlabel('Lambda')
    plt.ylabel('Error')

    plt.xticks([i / 10 for i in range(1, 11)])

    plt.show()


"""
learning plot - note that this required modification of sarsa lambda to return errors as well

monte_carlo_q_values, monte_carlo_policy = monte_carlo({}, 100000)
lambdas = []
for lmbd in [0, 1]:
    q_values, policy, errors = sarsa_lambda({}, lmbd, 1000)
    plt.plot(errors, label=f"lambda={lmbd}")
    plt.xlabel("Episode")
    plt.ylabel("Error")
plt.legend()

plt.savefig(f"learning.png")

"""
