# L1 - Intro to RL

### What makes reinforcement learning different from other machine learning paradigms?
- There is no supervisor, only a reward signal
- Feedback is delayed, not instantaneous
- Time really matters (sequential, non i.i.d data)
- Agent’s actions affect the subsequent data it receives

### What is Reward
- A reward Rt is a scalar feedback signal
- Indicates how well agent is doing at step *t*
- The agent’s job is to maximise cumulative reward
- Reinforcement learning is based on the **reward hypothesis** (All goals can be described by the maximisation of expected cumulative reward)

### Agent and Environment
<img src="../images/agent_and_env.png" alt="Agent and Environment" width="50%">

### History of State
- **History** is the sequence of observations, actions and rewards up to time t
  - Ht = O1, R1, A1, ..., At−1, Ot, Rt
- **State** is the info used to determine what happens next, is a function of history:
  - St = f(Ht)
