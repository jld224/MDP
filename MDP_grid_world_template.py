# AI Project 2 - MDP
# Grid world - part 1 policy evaluation; part 2 value iteration
import pandas as pd
import numpy as np
import sys


def policyEvaluation(policy, reward, gamma, duration):
    val = np.zeros(policy.shape)
    value_grid = np.zeros_like(val)
    terminal_states = [(4, 3), (4, 2)]
    iter_count = 0

    while iter_count < duration:
        new_val = np.copy(value_grid)
        for i in range(val.shape[0]):
            for j in range(val.shape[1]):
                if (i, j) in terminal_states:
                    continue
                action = policy[i, j]
                if action == 1:  # up
                    next_vals = []
                    if i > 0: next_vals.append(0.8 * (reward + gamma * value_grid[i - 1, j]))
                    if j > 0: next_vals.append(0.1 * (reward + gamma * value_grid[i, j - 1]))
                    if j < val.shape[1] - 1: next_vals.append(0.1 * (reward + gamma * value_grid[i, j + 1]))
                    new_val[i, j] = sum(next_vals)
                # .. do the similar for case action -1, 2, -2 (down, right, left)

        value_grid = new_val
        iter_count += 1

    return value_grid


def valueIteration(reward, gamma, prob, duration):
    value_grid = np.zeros((3, 4))
    policy = np.zeros((3, 4))
    terminal_states = [(4, 3), (4, 2)]
    actions = {'up': (1, 0), 'right': (0, 1), 'down': (-1, 0), 'left': (0, -1)}
    iter_count = 0

    while iter_count < duration:
        new_value_grid = np.copy(value_grid)
        for i in range(3):
            for j in range(4):
                if (i, j) in terminal_states:
                    continue
                val = []
                for action in actions.keys():
                    next_step = (i + actions[action][0], j + actions[action][1])
                    if (next_step[0] in range(3)) and (next_step[1] in range(4)):
                        val.append((reward + prob * gamma * value_grid[next_step]))
                new_value_grid[i, j] = max(val)
                policy[i, j] = np.argmax(val) + 1
        value_grid = new_value_grid
        iter_count += 1

    return policy


# No need to change the main function.
def main():
    part, reward, arg3 = sys.argv[1:]
    gamma = 0.95
    duration = 50
    if part == "1":
        policyFileName = arg3
        policyData = pd.read_csv(policyFileName, header=None)
        policy = policyData.to_numpy(dtype=int)
        policy = policy[::-1]  # flip the rows to match the setup
        values = policyEvaluation(policy, float(reward), gamma, duration)
        print("The expected utility of policy given in " + policyFileName +
              " after", duration, "iterations :")
        print(values[::-1])  # flip the rows to match the setup
    elif part == "2":
        prob = float(arg3)
        print("Optimal policy after", duration, "iterations :")
        policy = valueIteration(float(reward), gamma, prob, duration)
        print(policy[::-1])  # flip the rows to match the setup
    else:
        print("arg error")

    print("\ndone!")


if __name__ == '__main__':
    main()
