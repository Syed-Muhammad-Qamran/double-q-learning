# 🎯 Double Q-Learning & Eliminating Maximization Bias

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of **Double Q-Learning** (Hado van Hasselt, 2010) solving the classic **Maximization Bias MDP** from Richard Sutton and Andrew Barto's *Reinforcement Learning: An Introduction* (Example 6.7). Features an interactive **Streamlit** dashboard replicating the iconic Figure 6.5 learning curves and demonstrating how decoupling action selection from evaluation cures overestimation.

---

## 📌 Problem Overview: The Maximization Bias MDP

```text
                  [ TERMINAL ]
                       ▲
                       │ Action: RIGHT (Reward = 0)
                       │
                  [  State A  ] (Start)
                       │
                       │ Action: LEFT (Reward = 0)
                       ▼
                  [  State B  ] ──(10 Noisy Actions)──> [ TERMINAL ]
                                  Rewards ~ N(-0.1, 1.0)
```

* **State A:** 
  * `Right`: Terminates with a guaranteed reward of $0.0$. *(Optimal choice!)*
  * `Left`: Transitions to State B with a reward of $0.0$.
* **State B:**
  * Has $10$ actions, each terminating with reward $R \sim \mathcal{N}(-0.1, 1.0)$.
  * True expected return: $-0.10$.

### Why Standard Q-Learning Fails:
Because of the $\max_a Q(S', a)$ operator, standard Q-Learning takes the maximum over 10 noisy actions. By random chance, at least one action rolls a positive reward ($+1.5$ to $+2.0$). Standard Q-Learning latches onto this outlier, overestimates the value of State B, and **chooses Left over 75% of the time**!

---

## 🧠 The Double Q-Learning Solution

Double Q-Learning maintains **two independent Q-tables ($Q_1$ and $Q_2$)**:
1. **$Q_1$ selects the best action:** $A^* = \arg\max_a Q_1(S', a)$
2. **$Q_2$ evaluates its true value:** $\text{Target} = R + \gamma Q_2(S', A^*)$

With a 50% probability at each step, the roles of $Q_1$ and $Q_2$ are swapped.

Because $Q_2$ is independent, its evaluation of $A^*$ has expected value $-0.10$. The maximization bias drops to zero, and Double Q-Learning correctly chooses **Right**!

---

## 🚀 Features

* **Sutton & Barto Fig 6.5 Replication:** Live curve showing the percentage of Left actions taken from State A across 100+ averaged Monte Carlo runs.
* **Overestimation Trajectory:** Compare the estimated value of $Q(A, \text{Left})$ over time against the true mean ($-0.10$).
* **Single-Step Casino Simulator:** Sample live rewards from State B to visually observe how $\max$ gets fooled and how Double Q filters out the outlier.
* **Deep RL Connection:** Explains how this foundation led to DeepMind's **Double DQN (DDQN)**.

---

## 💻 Quickstart & Running

### 1. Clone & Install
```bash
git clone https://github.com/Syed-Muhammad-Qamran/double-q-learning.git
cd double-q-learning
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 👤 Author
* **Syed Muhammad Qamran**
* GitHub: [@Syed-Muhammad-Qamran](https://github.com/Syed-Muhammad-Qamran)
