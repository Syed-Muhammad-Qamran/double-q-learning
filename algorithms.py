import numpy as np

class StandardQLearning:
    def __init__(self, n_b_actions=10, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.n_b_actions = n_b_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q(A) has 2 actions (0: Left, 1: Right)
        self.q_a = np.zeros(2, dtype=np.float64)
        # Q(B) has n_b_actions actions
        self.q_b = np.zeros(n_b_actions, dtype=np.float64)

    def select_action(self, state):
        if state == 0:  # State A
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.randint(2)
            max_q = np.max(self.q_a)
            best_actions = np.where(self.q_a == max_q)[0]
            return int(np.random.choice(best_actions))
        else:           # State B
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.randint(self.n_b_actions)
            max_q = np.max(self.q_b)
            best_actions = np.where(self.q_b == max_q)[0]
            return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state):
        if state == 0:  # State A
            if next_state == 1:  # Left to B
                best_next_q = np.max(self.q_b)
            else:                # Right to Terminal
                best_next_q = 0.0
            td_target = reward + self.gamma * best_next_q
            self.q_a[action] += self.alpha * (td_target - self.q_a[action])
        elif state == 1:  # State B -> Terminal
            td_target = reward  # Terminal state has value 0
            self.q_b[action] += self.alpha * (td_target - self.q_b[action])


class DoubleQLearning:
    def __init__(self, n_b_actions=10, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.n_b_actions = n_b_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Two independent Q-tables
        self.q1_a = np.zeros(2, dtype=np.float64)
        self.q2_a = np.zeros(2, dtype=np.float64)
        self.q1_b = np.zeros(n_b_actions, dtype=np.float64)
        self.q2_b = np.zeros(n_b_actions, dtype=np.float64)

    def select_action(self, state):
        if state == 0:  # State A
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.randint(2)
            avg_q = (self.q1_a + self.q2_a) / 2.0
            max_q = np.max(avg_q)
            best_actions = np.where(avg_q == max_q)[0]
            return int(np.random.choice(best_actions))
        else:           # State B
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.randint(self.n_b_actions)
            avg_q = (self.q1_b + self.q2_b) / 2.0
            max_q = np.max(avg_q)
            best_actions = np.where(avg_q == max_q)[0]
            return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state):
        # 50% chance to update Q1, 50% chance to update Q2
        if np.random.uniform(0, 1) < 0.5:
            # Update Q1 using Q2 as the evaluator
            if state == 0:
                if next_state == 1:
                    # Q1 chooses action in B, Q2 evaluates it
                    best_a = int(np.random.choice(np.where(self.q1_b == np.max(self.q1_b))[0]))
                    best_next_q = self.q2_b[best_a]
                else:
                    best_next_q = 0.0
                td_target = reward + self.gamma * best_next_q
                self.q1_a[action] += self.alpha * (td_target - self.q1_a[action])
            elif state == 1:
                td_target = reward
                self.q1_b[action] += self.alpha * (td_target - self.q1_b[action])
        else:
            # Update Q2 using Q1 as the evaluator
            if state == 0:
                if next_state == 1:
                    # Q2 chooses action in B, Q1 evaluates it
                    best_a = int(np.random.choice(np.where(self.q2_b == np.max(self.q2_b))[0]))
                    best_next_q = self.q1_b[best_a]
                else:
                    best_next_q = 0.0
                td_target = reward + self.gamma * best_next_q
                self.q2_a[action] += self.alpha * (td_target - self.q2_a[action])
            elif state == 1:
                td_target = reward
                self.q2_b[action] += self.alpha * (td_target - self.q2_b[action])


def run_benchmark(n_runs=100, n_episodes=300, n_b_actions=10, alpha=0.1, gamma=1.0, epsilon=0.1):
    """
    Simulates n_runs of the Maximization Bias experiment (Sutton & Barto Fig 6.5).
    Returns the percentage of Left actions taken from State A at each episode index.
    """
    from env import MaximizationBiasMDP
    
    q_left_counts = np.zeros(n_episodes, dtype=np.float64)
    dq_left_counts = np.zeros(n_episodes, dtype=np.float64)
    
    q_value_left = np.zeros(n_episodes, dtype=np.float64)
    dq_value_left = np.zeros(n_episodes, dtype=np.float64)

    for run in range(n_runs):
        env = MaximizationBiasMDP(n_b_actions=n_b_actions)
        q_agent = StandardQLearning(n_b_actions=n_b_actions, alpha=alpha, gamma=gamma, epsilon=epsilon)
        dq_agent = DoubleQLearning(n_b_actions=n_b_actions, alpha=alpha, gamma=gamma, epsilon=epsilon)

        # Standard Q-learning run
        for ep in range(n_episodes):
            s = env.reset()
            a = q_agent.select_action(s)
            if a == 0:  # Action Left
                q_left_counts[ep] += 1
            
            ns, r, done = env.step(a)
            q_agent.update(s, a, r, ns)
            if not done:
                s = ns
                a = q_agent.select_action(s)
                ns, r, done = env.step(a)
                q_agent.update(s, a, r, ns)
            q_value_left[ep] += q_agent.q_a[0]

        # Double Q-learning run
        for ep in range(n_episodes):
            s = env.reset()
            a = dq_agent.select_action(s)
            if a == 0:  # Action Left
                dq_left_counts[ep] += 1
                
            ns, r, done = env.step(a)
            dq_agent.update(s, a, r, ns)
            if not done:
                s = ns
                a = dq_agent.select_action(s)
                ns, r, done = env.step(a)
                dq_agent.update(s, a, r, ns)
            dq_value_left[ep] += (dq_agent.q1_a[0] + dq_agent.q2_a[0]) / 2.0

    pct_q_left = (q_left_counts / n_runs) * 100.0
    pct_dq_left = (dq_left_counts / n_runs) * 100.0
    
    avg_q_left = q_value_left / n_runs
    avg_dq_left = dq_value_left / n_runs

    return {
        "pct_q_left": pct_q_left,
        "pct_dq_left": pct_dq_left,
        "avg_q_left": avg_q_left,
        "avg_dq_left": avg_dq_left
    }
