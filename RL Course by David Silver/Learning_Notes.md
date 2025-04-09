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
