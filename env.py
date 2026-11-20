import numpy as np

class MaximizationBiasMDP:
    """
    Sutton & Barto (Example 6.7) Maximization Bias MDP.
    States:
        0: State A (Start)
        1: State B
        2: Terminal
    From State A:
        Action 0: Left -> goes to State B with Reward = 0
        Action 1: Right -> goes to Terminal with Reward = 0
    From State B:
        n_b_actions actions: each goes to Terminal with Reward ~ N(mean, std)
    """
    STATE_A = 0
    STATE_B = 1
    TERMINAL = 2

    def __init__(self, n_b_actions=10, mean=-0.1, std=1.0):
        self.n_b_actions = n_b_actions
        self.mean = mean
        self.std = std
        self.state = self.STATE_A

    def reset(self):
        self.state = self.STATE_A
        return self.state

    def step(self, action):
        if self.state == self.STATE_A:
            if action == 0:  # Left to B
                self.state = self.STATE_B
                return self.state, 0.0, False
            else:            # Right to Terminal
                self.state = self.TERMINAL
                return self.state, 0.0, True

        elif self.state == self.STATE_B:
            # Any action in B gives reward ~ N(mean, std) and terminates
            reward = np.random.normal(self.mean, self.std)
            self.state = self.TERMINAL
            return self.state, reward, True

        else:
            return self.TERMINAL, 0.0, True
