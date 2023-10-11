## MDP Grid World Project

This AI project involves a grid world where an agent makes decisions based on two methods: policy evaluation and value iteration. The goal is to determine the best policies for actions in a grid world format.

### Getting started

The project was designed and tested using Python 3. It is required to run the project.

### Libraries used

- Pandas
- Numpy
- sys

### Project Files

- `MDPGrid.py`: This Python file contains the main program for policy evaluation and value iteration. 

### Functions

The main functionalities of the program are separated into multiple parts.

- `policyEvaluation(policy, reward, gamma, duration)`: This function evaluates a given policy through time and returns the expected utility of the policy after the specified number of iterations.
  
- `valueIteration(reward, gamma, prob, duration)`: This function does value iteration to find the optimal policy. It returns the optimal policy.

- `main()`: This is the main driver function of the code that calculates and displays the expected utility and the optimal policy.

### How to run the project

Navigate to the directory containing the file `MDPGrid.py`. Then run:

For policy evaluation:

```
python3 MDPGrid.py 1 [reward] [policy_file.csv]
```

For value iteration:

```
python3 MDPGrid.py 2 [reward] [probability]
```

Where:
- `[reward]` is the reward to states, other than the two terminal states
- `[probability]` is the transition probability to the intended direction
- `<policy_file.csv>` is the filename of the policy expressed as a .csv file

Examples:

- Policy evaluation with a reward of -0.04 and using `case1.csv` file for policy:
  
   `python3 MDPGrid.py 1 -0.04 case1.csv`
  
- Value iteration with a reward of -0.05 and a transition probability of 0.7.
  
   `python3 MDPGrid.py 2 -0.05 0.7`

Please replace `[reward]`, `[probability]`, and `<policy_file.csv>` with your actual values.

### Output

The results will be printed to the console. These will include the expected utility of a specified policy or the optimal policy after a specified number of iterations.
