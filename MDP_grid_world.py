# AI Project 2 - MDP
# Grid world - part 1 policy evaluation; part 2 value iteration
import pandas as pd
import numpy as np
import sys

def policyEvaluation(policy, reward, gamma, duration):
    print("policy:\n", policy, "\nreward, gamma, duration:\n", reward, gamma, duration)

    numRows, numCols = policy.shape
    val = np.zeros(policy.shape)  # utility values

    # initialize states
    for k in range(duration):
        temp_val = val.copy()

        for i in range(numRows):
            for j in range(numCols):

                # check for terminal states and blocked state
                if (i == 0 and j == 3) or (i == 1 and j == 3) or (i == 1 and j == 1):
                    continue

                # get directions based on policy
                if policy[i, j] == 1:  # up
                    up = i - 1 if i - 1 >= 0 else i
                    down = i + 1 if i + 1 < numRows else i
                    left = j - 1 if j - 1 >= 0 else j
                    right = j + 1 if j + 1 < numCols else j
                elif policy[i, j] == -1:  # down
                    down = i + 1 if i + 1 < numRows else i
                    up = i - 1 if i - 1 >= 0 else i
                    left = j - 1 if j - 1 >= 0 else j
                    right = j + 1 if j + 1 < numCols else j
                elif policy[i, j] == 2:  # right
                    right = j + 1 if j + 1 < numCols else j
                    left = j - 1 if j - 1 >= 0 else j
                    up = i - 1 if i - 1 >= 0 else i
                    down = i + 1 if i + 1 < numRows else i
                else:  # left
                    left = j - 1 if j - 1 >= 0 else j
                    right = j + 1 if j + 1 < numCols else j
                    up = i - 1 if i - 1 >= 0 else i
                    down = i + 1 if i + 1 < numRows else i

                # compute expected utility
                val[i][j] = reward + gamma * (
                            0.8 * temp_val[up, j] + 0.1 * temp_val[i, left] + 0.1 * temp_val[i, right])

    return val


def valueIteration(reward, gamma, prob, duration):
    print("\nreward, gamma, prob, duration:\n", reward, gamma, prob, duration)

    numRows, numCols = 3, 4  # grid shape
    policy = np.empty((numRows, numCols), dtype=object)  # initial policy
    val = np.zeros((numRows, numCols))  # initial values

    for k in range(duration):
        new_val = val.copy()

        for i in range(numRows):
            for j in range(numCols):

                # check for terminal states and blocked state
                if (i == 0 and j == 3) or (i == 1 and j == 3) or (i == 1 and j == 1):
                    continue

                directions = {'up': [i - 1 if i - 1 >= 0 else i, j],
                              'left': [i, j - 1 if j - 1 >= 0 else j],
                              'down': [i + 1 if i + 1 < numRows else i, j],
                              'right': [i, j + 1 if j + 1 < numCols else j]}

                rewards = []
                for direction, [di, dj] in directions.items():
                    r = reward + gamma * (prob * new_val[di, dj] +
                                          (1 - prob) / 2 * new_val[directions[
                                'right' if direction == 'left' else 'left' if direction == 'down' else 'down' if direction == 'right' else 'up'][
                                0],
                            directions[
                                'right' if direction == 'left' else 'left' if direction == 'down' else 'down' if direction == 'right' else 'up'][
                                1]] +
                                          (1 - prob) / 2 * new_val[directions[
                                'left' if direction == 'right' else 'right' if direction == 'up' else 'up' if direction == 'down' else 'down'][
                                0],
                            directions[
                                'left' if direction == 'right' else 'right' if direction == 'up' else 'up' if direction == 'down' else 'down'][
                                1]])
                    rewards.append(r)

                val[i, j] = max(rewards)
                policy[i, j] = ['up', 'left', 'down', 'right'][np.argmax(rewards)]

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
        print(values)  # flip the rows to match the setup
    elif part == "2":
        prob = float(arg3)
        print("Optimal policy after", duration, "iterations :")
        policy = valueIteration(float(reward), gamma, prob, duration)
        print(policy)  # flip the rows to match the setup
    else:
        print("arg error")

    print("\ndone!")


if __name__ == '__main__':
    main()
