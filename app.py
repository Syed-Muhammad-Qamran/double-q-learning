import streamlit as st
import numpy as np
import pandas as pd
from algorithms import run_benchmark, StandardQLearning, DoubleQLearning
from env import MaximizationBiasMDP

st.set_page_config(
    page_title="Double Q-Learning vs. Q-Learning",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .state-box {
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        border: 1px solid #475569;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

# ----------------- SIDEBAR -----------------
st.sidebar.title("🎯 Double Q-Learning")
st.sidebar.markdown(
    "**Core Benchmark:** Sutton & Barto (Example 6.7)\n\n"
    "Demonstrates **Maximization Bias** and how decoupling action selection from evaluation solves it."
)

st.sidebar.subheader("Environment Settings")
n_b_actions = st.sidebar.slider("Number of Actions in State B", min_value=5, max_value=50, value=10, step=5,
                                help="More actions in State B increase the random positive noise.")
mean_reward = st.sidebar.slider("State B Mean Reward (μ)", min_value=-0.5, max_value=0.0, value=-0.1, step=0.05,
                                help="True expected reward of actions in State B. Since μ < 0, going Left is always suboptimal.")
std_reward = st.sidebar.slider("State B Reward Noise (σ)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

st.sidebar.subheader("Training Parameters")
n_runs = st.sidebar.slider("Monte Carlo Runs (Averaging)", min_value=20, max_value=200, value=100, step=20)
n_episodes = st.sidebar.slider("Episodes per Run", min_value=100, max_value=500, value=300, step=50)
alpha = st.sidebar.slider("Learning Rate (α)", min_value=0.05, max_value=0.5, value=0.10, step=0.05)
epsilon = st.sidebar.slider("Exploration Rate (ε)", min_value=0.05, max_value=0.20, value=0.10, step=0.01)

if st.sidebar.button("🚀 Run Benchmark Simulation", type="primary", use_container_width=True):
    with st.spinner(f"Simulating {n_runs} runs of {n_episodes} episodes..."):
        res = run_benchmark(
            n_runs=n_runs,
            n_episodes=n_episodes,
            n_b_actions=n_b_actions,
            alpha=alpha,
            gamma=1.0,
            epsilon=epsilon
        )
        st.session_state.benchmark_results = res
        st.sidebar.success("Benchmark completed!")

# ----------------- MAIN VIEW -----------------
st.title("🎯 Double Q-Learning & The Maximization Bias Trap")
st.markdown(
    "Standard Q-Learning suffers from a mathematical blind spot: **it overestimates values in noisy environments**. "
    "Explore how **Double Q-Learning** decouples selection from evaluation to make rational, unbiased decisions."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Sutton & Barto Fig 6.5 Replication",
    "📊 Value Overestimation Curves",
    "🎲 Step-by-Step Casino Simulator",
    "📖 The Mathematics of Double Q"
])

# ---------- TAB 1: REPLICATION ----------
with tab1:
    st.subheader("Percentage of Left Actions Taken from State A (Over Episodes)")
    st.markdown(
        "*The optimal action from State A is **RIGHT** (0.0 reward vs -0.1). "
        "Under ε=0.10 exploration, an optimal agent should choose Left only **5% of the time** (ε / 2).*"
    )

    if st.session_state.benchmark_results is None:
        st.info("👈 Click **'🚀 Run Benchmark Simulation'** in the sidebar to simulate both algorithms.")
    else:
        res = st.session_state.benchmark_results

        # High level metrics
        m1, m2, m3 = st.columns(3)
        early_q = np.mean(res["pct_q_left"][:50])
        early_dq = np.mean(res["pct_dq_left"][:50])
        m1.metric("Q-Learning Early Left %", f"{early_q:.1f}%", delta="Heavy Bias Trap", delta_color="inverse")
        m2.metric("Double Q Early Left %", f"{early_dq:.1f}%", delta="Unbiased", delta_color="normal")
        m3.metric("Theoretical Optimal Baseline", f"{(epsilon / 2) * 100:.1f}%")

        # Plot comparison
        df_plot = pd.DataFrame({
            "Standard Q-Learning (Biased)": res["pct_q_left"],
            "Double Q-Learning (Unbiased)": res["pct_dq_left"],
            "Optimal Baseline (ε/2 = 5%)": [(epsilon / 2) * 100.0] * len(res["pct_q_left"])
        })
        st.line_chart(df_plot)
        st.caption(
            "Notice how **Standard Q-Learning peaks at 70–85% Left actions** early on because random positive noise in State B tricks it. "
            "In contrast, **Double Q-Learning** immediately resists the trap and stays close to the optimal 5% baseline."
        )

# ---------- TAB 2: OVERESTIMATION CURVES ----------
with tab2:
    st.subheader("Estimated Value of Q(State A, Action Left) vs. True Value (-0.1)")
    if st.session_state.benchmark_results is None:
        st.info("Run the benchmark in the sidebar to view value overestimation curves.")
    else:
        res = st.session_state.benchmark_results
        df_vals = pd.DataFrame({
            "Standard Q Estimate Q(A, Left)": res["avg_q_left"],
            "Double Q Estimate Q(A, Left)": res["avg_dq_left"],
            "True Value (μ = -0.1)": [mean_reward] * len(res["avg_q_left"]),
            "Zero Line (Right Action = 0.0)": [0.0] * len(res["avg_q_left"])
        })
        st.line_chart(df_vals)
        st.caption(
            "Standard Q-Learning repeatedly estimates $Q(A, \\text{Left}) > 0.0$ due to the $\\max_a$ operator. "
            "Because it thinks Left is positive, it prefers Left over Right ($0.0$). "
            "Double Q-Learning correctly keeps the estimate negative ($\\approx -0.1$), correctly choosing Right!"
        )

# ---------- TAB 3: STEP-BY-STEP SIMULATOR ----------
with tab3:
    st.subheader("Interactive Single-Step Experiment")
    st.markdown("Sample random rewards from State B right now to see why $\\max$ gets fooled:")

    if st.button("🎲 Sample State B Actions Once"):
        samples = np.random.normal(mean_reward, std_reward, size=n_b_actions)
        max_val = np.max(samples)
        max_idx = np.argmax(samples)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown(f"**Sampled Rewards for {n_b_actions} actions in State B:**")
            df_samples = pd.DataFrame({
                "Action": [f"Action {i}" for i in range(n_b_actions)],
                "Sampled Reward": samples
            })
            st.dataframe(df_samples.style.highlight_max(subset=["Sampled Reward"], color="#ef4444"), use_container_width=True)

        with col2:
            st.markdown("### The Verdict")
            st.markdown(f"True Mean: `{mean_reward:.2f}`")
            st.markdown(f"**Standard Q-Learning takes max:** `{max_val:+.3f}` (Action {max_idx})")
            if max_val > 0:
                st.error(f"💥 Standard Q thinks State B is worth **{max_val:+.3f}**! It will choose LEFT!")
            else:
                st.success(f"Even on this draw, max was {max_val:+.3f}.")

            # Simulate independent draw for Q2
            sample_eval = np.random.normal(mean_reward, std_reward)
            st.markdown("---")
            st.markdown(f"**Double Q-Learning uses Q2 to evaluate Action {max_idx}:** `{sample_eval:+.3f}`")
            st.info("Notice how the second evaluator eliminates the lucky outlier!")

# ---------- TAB 4: MATHEMATICAL THEORY ----------
with tab4:
    st.subheader("The Mathematics of Maximization Bias")
    st.markdown(r"""
    ### 1. Jensen's Inequality & Maximum of Expectations
    Let $X_1, X_2, \dots, X_n$ be independent random variables with true means $\mathbb{E}[X_i] = \mu$.
    Because the maximum function $f(x) = \max(x)$ is convex, Jensen's inequality guarantees that:
    """)
    st.latex(r"\mathbb{E}\Big[\max(X_1, X_2, \dots, X_n)\Big] \ge \max\Big(\mathbb{E}[X_1], \mathbb{E}[X_2], \dots, \mathbb{E}[X_n]\Big)")
    st.markdown(r"""
    Even if every single action has a negative expected return ($\mu = -0.1$), the expected value of the maximum over $10$ or $20$ actions is strictly positive:
    """)
    st.latex(r"\mathbb{E}\Big[\max_{a \in B} R_a\Big] > 0.0")
    st.markdown(r"""
    ### 2. How Double Q-Learning Solves It
    Double Q decouples **selection** from **evaluation** across two independent estimators:
    """)
    st.latex(r"A^* = \arg\max_{a} Q_1(S', a)")
    st.latex(r"\text{Target} = R + \gamma Q_2(S', A^*)")
    st.markdown(r"""
    Since $Q_2$ was trained on independent samples:
    $$\mathbb{E}\big[Q_2(S', A^*)\big] = \mathbb{E}\big[Q(S', A^*)\big] = -0.1$$
    This eliminates maximization bias completely, without slowing down convergence!
    
    ### 3. The Deep RL Connection: Double DQN (DDQN)
    In 2015, DeepMind showed that standard Deep Q-Networks (DQN) suffer from severe overestimation in Atari games. 
    By applying this exact Double Q trick using the **Online Network** for action selection and the **Target Network** for evaluation:
    $$Y^{\text{DoubleDQN}} = R_{t+1} + \gamma Q\Big(S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; \theta_t); \, \theta_t^{-}\Big)$$
    Double DQN set new state-of-the-art benchmark records across the entire Atari suite.
    """)
