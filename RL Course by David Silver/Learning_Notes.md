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

### Environment State
<img src="../images/env_state.png" alt="Agent and Environment" width="50%">

### Agent State
<img src="../images/agent_state.png" alt="Agent and Environment" width="50%">

### Information State
**Information State is Markov State.** The environment state is Markov. The history Ht is Markov.
<img src="../images/info_state.png" alt="Agent and Environment" width="50%">

### Fully Observable Environments
- Agent state (S<sup>a</sup><sub>t</sub>) = environment state (S<sup>e</sup><sub>t</sub>) = information state
- Formally, this is a **Markov decision process** (MDP)

### Partially Observable Environments
-  agent state (S<sup>a</sup><sub>t</sub>) != environment state (S<sup>e</sup><sub>t</sub>)
-  Formally this is a **partially observable Markov decision process** (POMDP)
<img src="../images/partial_MDP.png" alt="Agent and Environment" width="50%">

### Major Components of an RL Agent
- An RL agent may include **one or more** of these components:
  - Policy: agent’s behaviour function (map from state to action)
    - <img src="../images/policy.png" alt="Agent and Environment" width="50%">
  - Value function: how good is each state and/or action, it is a prediction of future reward
  - Model: agent’s representation of the environment. It predicts what the environment will do next, including state transition matrix and reward.
    - <img src="../images/model.png" alt="Agent and Environment" width="50%">

### Categorizing RL Agents
- Based on Policy and Value Function
  - <img src="../images/categorizing_RL_agent_1.png" alt="Agent and Environment" width="30%">
- Based on Model
  - <img src="../images/categorizing_RL_agent_2.png" alt="Agent and Environment" width="30%">
- Overall
  - <img src="../images/categorizing_RL_agent_3.png" alt="Agent and Environment" width="30%">

### Learning and Planning
Two fundamental problems in sequential decision making
- Reinforcement Learning:
  - The environment is initially unknown
  - The agent interacts with the environment
  - The agent improves its policy
- Planning:
  - A model of the environment is known
  - The agent performs computations with its model (without any external interaction)
  - The agent improves its policy
  - a.k.a. deliberation, reasoning, introspection, pondering, thought, search

- Example: RL
  - <img src="../images/exp_RL.png" alt="Agent and Environment" width="50%">
- Example: Planning
  - <img src="../images/exp_planning.png" alt="Agent and Environment" width="50%">

### Exploration and Exploitation
- Exploration finds more information about the environment
- Exploitation exploits known information to maximise reward
- It is usually important to explore as well as exploit

### Prediction and Control in Learning
- Prediction: evaluate the future
  - Given a policy, find the value function
- Control: optimise the future
  - Find the best policy
  - What is the optimal value function over all possible policies?
  - What is the optimal policy?


# L2 - Markov Decision Processes

Markov decision processes formally describe an *fully observable* environment for reinforcement learning.

Almost all RL problems can be formalised as MDPs, e.g.
- Optimal control primarily deals with continuous MDPs
- Partially observable problems can be converted into MDPs
- Bandits are MDPs with one state

### Markov Property
The future is independent of the past given the present
- <img src="../images/markov_property.png" alt="Agent and Environment" width="50%">

### State Transition Matrix
- <img src="../images/state_transition_matrix.png" alt="Agent and Environment" width="50%">

### Markov Process
A Markov process is a **memoryless random process**, i.e. a sequence of random states S1, S2, ... with the Markov property.
- <img src="../images/markov_process.png" alt="Agent and Environment" width="40%">

### Markov Reward Process
A Markov reward process is a Markov chain with values.
- <img src="../images/markov_reward_process.png" alt="Agent and Environment" width="40%">

### Return
<img src="../images/return.png" alt="Agent and Environment" width="50%">

### Why discount?
- Mathematically convenient to discount rewards
- Avoids infinite returns in cyclic Markov processes
- Uncertainty about the future may not be fully represented
- If the reward is financial, immediate rewards may earn more interest than delayed rewards
- Animal/human behaviour shows preference for immediate reward
- It is sometimes possible to use undiscounted Markov reward processes (i.e. γ = 1), e.g. if all sequences terminate

### Value Function
- The state value function v(s) of an MRP is the expected return starting from state s
- **v(s) = E [Gt | St = s]**

### Bellman Equation for MRPs
<img src="../images/bellman_equation_for_MRPs.png" alt="Agent and Environment" width="40%">

- Bellman Equation in Matrix Form
- <img src="../images/bellman_equation_matrix_form.png" alt="Agent and Environment" width="40%">

- The Bellman equation is a linear equation and can be solved directly.
- Computational complexity is O(n3) for n states
- Direct solution only possible for small MRPs
- There are many iterative methods for large MRPs, e.g.
  - Dynamic programming
  - Monte-Carlo evaluation
  - Temporal-Difference learning

### Markov Decision Process
- A Markov decision process (MDP) is a Markov reward process with decisions (action).
- <img src="../images/mdp.png" alt="Agent and Environment" width="40%">

- Summary:
  - Markov Process (MP): {S, P}
  - Markov Reward Process (MRP): {S, P, R, γ}
  - Markov Decision Process (MDP): {S, A, P, R, γ}, **policy decide with action to take, after that, model (P) decides what the next state is**

### Policies
- A policy π is a distribution over actions given states,
- **π(a|s) = P[At = a | St = s]**
- MDP policies depend on the current state (not the history)
- Policies are stationary (time-independent)
- Given an MDP M = (S, A, P, R, γ) and a policy π
- The state sequence S1, S2, ... is a Markov process (S,P<sup>π</sup>)
- The state and reward sequence S1, R2, S2, ... is a Markov reward process (S,P<sup>π</sup>,R<sup>π</sup>,γ)
  - <img src="../images/MDP_policy_P_R.png" alt="Agent and Environment" width="20%">

#### Policy State Value Function
- **v<sub>π</sub>(s) = E<sub>π</sub>[Gt|St = s]**
- **v<sub>π</sub>(s) = E<sub>π</sub>[R<sub>t+1</sub> + γv<sub>π</sub>(S<sub>t+1</sub>) | St = s]**
- <img src="../images/state_value_function_to_action_value_function.png" alt="Agent and Environment" width="20%">
- <img src="../images/state_value_function_recur.png" alt="Agent and Environment" width="30%">
- **v<sub>π</sub> = R<sub>π</sub> + γP<sub>π</sub>v<sub>π</sub>**

#### Policy Action Value Function
- **q<sub>π</sub>(s,a) = E<sub>π</sub>[Gt|St = s, At = a]**
- **q<sub>π</sub>(s,a) = E<sub>π</sub>[R<sub>t+1</sub> + γq<sub>π</sub>(S<sub>t+1</sub>,A<sub>t+1</sub>) | St = s, At = a]**
- <img src="../images/action_value_function.png" alt="Agent and Environment" width="25%">
- <img src="../images/action_value_function_recur.png" alt="Agent and Environment" width="35%">

#### Optimal Value Function
<img src="../images/optimal_value_function.png" alt="Agent and Environment" width="45%">

#### Optimal Policy
<img src="../images/optimal_policy.png" alt="Agent and Environment" width="45%">

#### Bellman Optimality Equation for V* and Q*
<img src="../images/bellman_optimal_value_equation.png" alt="Agent and Environment" width="45%">

#### Solving the Bellman Optimality Equation
<img src="../images/solving_bellman_optimal_equation.png" alt="Agent and Environment" width="30%">

### Partially observable MDPs (POMDPs)
<img src="../images/POMDPs.png" alt="Agent and Environment" width="45%">


# L3 - Planning by Dynamic Programming

### What is Dynamic Programming?
- **Dynamic** sequential or temporal component to the problem
- **Programming** optimising a “program”, i.e. a policy
- A method for solving complex problems
- By breaking them down into subproblems
  - Solve the subproblems
  - Combine solutions to subproblems
 
### Requirements for Dynamic Programming
- Dynamic Programming is a very general solution method for problems which have two properties:
- Optimal substructure
  - Principle of optimality applies
  - Optimal solution can be decomposed into subproblems
- Overlapping subproblems
  - Subproblems recur many times
  - Solutions can be cached and reused
- Markov decision processes satisfy both properties
  - Bellman equation gives recursive decomposition
  - Value function stores and reuses solutions

### Planning by Dynamic Programming
- <img src="../images/planning_by_DP.png" alt="Agent and Environment" width="45%">

### Other Applications of Dynamic Programming
- <img src="../images/other_application_of_DP.png" alt="Agent and Environment" width="45%">

### Iterative Policy Evaluation
- <img src="../images/iterative_policy_evaluation1.png" alt="Agent and Environment" width="45%">
- <img src="../images/iterative_policy_evaluation.png" alt="Agent and Environment" width="40%">

### How to Improve a Policy
- <img src="../images/improve_a_policy.png" alt="Agent and Environment" width="45%">
- This process of **policy iteration** always converges to π∗

### Modified Policy Iteration
- No need to wait for policy evaluation to converge to vπ
- Can update policy every iteration (equivalent to value iteration)

### Generalised Policy Iteration
- <img src="../images/generalized_policy_iteration.png" alt="Agent and Environment" width="45%">

### Principle of Optimality
- <img src="../images/principle_of_optimality.png" alt="Agent and Environment" width="45%">

### Deterministic Value Iteration
- <img src="../images/deterministic_value_iteration.png" alt="Agent and Environment" width="45%">

### Value Iteration
- Unlike policy iteration, there is no explicit policy
- Intermediate value functions may not correspond to any policy
- Use **Bellman optimality backup** for each iteration (policy iteration use Bellman Expectation backup)
- The rest is the same as policy iteration
- <img src="../images/value_iteration.png" alt="Agent and Environment" width="45%">
- <img src="../images/value_iteration1.png" alt="Agent and Environment" width="45%">

### Synchronous Dynamic Programming Algorithms
- <img src="../images/synchronous_DP_algo.png" alt="Agent and Environment" width="45%">

### Asynchronous Dynamic Programming
- **All** states are backed up in parallel in **synchronous** backups
- **Asynchronous** DP backs up states individually, in any order
- For each selected state, apply the appropriate backup
- Can significantly reduce computation
- Guaranteed to converge if all states continue to be selected
- Three simple ideas for asynchronous dynamic programming:
  - In-place dynamic programming
  - Prioritised sweeping
  - Real-time dynamic programming

### In-Place Dynamic Programming
- <img src="../images/inplace_DP.png" alt="Agent and Environment" width="45%">

### Prioritised Sweeping
- <img src="../images/prioritised_sweeping.png" alt="Agent and Environment" width="45%">

### Real-Time Dynamic Programming
- <img src="../images/realtime_DP.png" alt="Agent and Environment" width="45%">

### Limitation of DP
- For each backup, every successor state and action is considered by using knowledge of MDP transiiton matrix and reward function
- It is only effective for medium-sized problems but **not for large problems**.
- For large problem, need to use **sample backup** (using sample rewards and sample transitions)
- Advantage of Sample Backup
  - Model-free: no advance knowledge of MDP required
  - Breaks the curse of dimensionality through sampling
  - Cost of backup is constant, independent of n = |S|

### Approximate Dynamic Programming
- <img src="../images/approximate_DP.png" alt="Agent and Environment" width="45%">

### Contraction Mapping Theorem
- <img src="../images/contraction_mapping_theorem.png" alt="Agent and Environment" width="45%">

### Proof of Convergence using Contraction Mapping Theorem
- Bellman Expectation Backup is a Contraction
- <img src="../images/bellman_expectation_backup_is_contraction.png" alt="Agent and Environment" width="45%">
- <img src="../images/convergence_of_policy_iteration.png" alt="Agent and Environment" width="45%">
- Bellman Optimality Backup is a Contraction
- <img src="../images/bellman_optimality_backup_is_contraction.png" alt="Agent and Environment" width="45%">
- <img src="../images/convergence_of_value_iteration.png" alt="Agent and Environment" width="45%">


# L4 - Model-Free Prediction

### Monte-Carlo Reinforcement Learning is Model-Free
- <img src="../images/mc_rl.png" alt="Agent and Environment" width="45%">

### MC Policy Evaluation
- <img src="../images/mc_policy_evaluation.png" alt="Agent and Environment" width="45%">

### Incremental Mean
- <img src="../images/incremental_mean.png" alt="Agent and Environment" width="45%">

### Incremental Monte-Carlo Updates
- <img src="../images/incremental_mc_update.png" alt="Agent and Environment" width="45%">


### Temporal-Difference Learning
- <img src="../images/td_learning.png" alt="Agent and Environment" width="45%">

### MC vs. TD0
- <img src="../images/mc_vs_td0.png" alt="Agent and Environment" width="45%">
- <img src="../images/mc_vs_td0_1.png" alt="Agent and Environment" width="45%">

### Advantages and Disadvantages of MC vs. TD
- <img src="../images/mc_td_compare.png" alt="Agent and Environment" width="45%">
- TD has higher bias but lower variance than MC
  - <img src="../images/mc_vs_td0_bias_variance_tradeoff.png" alt="Agent and Environment" width="45%">
  - <img src="../images/mc_vs_td0_bias_variance_tradeoff_1.png" alt="Agent and Environment" width="35%">
- Batch MC vs. Batch TD0 with AB Example
  - <img src="../images/AB_example.png" alt="Agent and Environment" width="45%">
  - <img src="../images/AB_example1.png" alt="Agent and Environment" width="45%">
- TD exploits Markov property
  - Usually more efficient in Markov environments
- MC does not exploit Markov property
  - Usually more effective in non-Markov environments

### MC, TD0 and DP Backup Comparison
- <img src="../images/mc_backup.png" alt="Agent and Environment" width="45%">
- <img src="../images/td0_backup.png" alt="Agent and Environment" width="45%">
- <img src="../images/dp_backup.png" alt="Agent and Environment" width="45%">

### Bootstrapping and Sampling
- <img src="../images/bootstrapping_and_sampling.png" alt="Agent and Environment" width="45%">
- <img src="../images/bootstrapping_and_sampling1.png" alt="Agent and Environment" width="45%">

### TD(lambda)
- <img src="../images/n_step_return.png" alt="Agent and Environment" width="45%">
- <img src="../images/lamda_return.png" alt="Agent and Environment" width="45%">
- <img src="../images/lamda_return1.png" alt="Agent and Environment" width="45%">

### Offline and Online RL
- Offline:
  - Update at the end of each episode, accumulate error at each step.
  - The total accumulated error at the end of episode will be the same for both forward-view and backward view.
- Online:
  - Update at each step of every episode
  - Forward-view and backward-view are slightly different, but exact online TD(lambda) (Sutton and von Seijen, ICML 2014) achieves perfect equivalence

### Eligibility Traces
- <img src="../images/eligibility_traces.png" alt="Agent and Environment" width="45%">

### Forward-View TD(lambda
- Do not use concept of Eligibility Traces
- <img src="../images/forward_view_td_lambda.png" alt="Agent and Environment" width="45%">

### Backward-View TD(lambda)
- Use concept of Eligibility Traces
- <img src="../images/backward_view_td_lambda.png" alt="Agent and Environment" width="45%">
- <img src="../images/backward_view_td_lambda1.png" alt="Agent and Environment" width="45%">

### Proof of TD(1) with Eligibility Trace is the same as MC Error
- <img src="../images/td1_error_same_as_mc_error.png" alt="Agent and Environment" width="45%">

### Proof of TD(lambda) with Eligibility Trace is the same as Lambda-Error
- <img src="../images/td_lambda_error_same_as_lambda_error.png" alt="Agent and Environment" width="45%">

### TD(lambda=0) is the same as TD0
- <img src="../images/td_lambda_and_td0.png" alt="Agent and Environment" width="45%">

### TD(lambda=1) is the same as MC in Offline Setting
- <img src="../images/td_lambda_and_mc.png" alt="Agent and Environment" width="45%">

### TD(lambda=1) and MC in Online Setting
- <img src="../images/mc_and_td1.png" alt="Agent and Environment" width="45%">

### TD(lambda) of Forward and Backward View in Online Setting
- <img src="../images/td_lambda_error_same_as_lambda_error.png" alt="Agent and Environment" width="45%">

### Summary of Forward and Backward TD(lambda)
- <img src="../images/summary_of_forward_and_backward_td_lambda.png" alt="Agent and Environment" width="45%">


# L5 - Model Free Control

### On-Policy vs. Off-Policy RL
- <img src="../images/on_policy_vs_off_policy_rl.png" alt="Agent and Environment" width="45%">

### Model-Free Policy Improvement Requires Value-Action Function, Not Value Function
- <img src="../images/greedy_policy_improvement_with_value_and_value_action_function.png" alt="Agent and Environment" width="45%">

### Need to do Exploration during policy Improvement - Epsilon-Greedy, cannot do Greedy Improvement
- <img src="../images/epsilon_greedy_exploration.png" alt="Agent and Environment" width="35%">
- <img src="../images/epsilon_greedy_policy_improvement.png" alt="Agent and Environment" width="35%">

### GLIE Monte-Carlo Control, On-Policy
- <img src="../images/MC_model_free_control.png" alt="Agent and Environment" width="35%">
- <img src="../images/GLIE_definition.png" alt="Agent and Environment" width="35%">
- <img src="../images/GLIE_MC_model_free_control.png" alt="Agent and Environment" width="35%">

### TD Model-Free Control, On-Policy
- <img src="../images/TD_model_free_control.png" alt="Agent and Environment" width="35%">

### On-Policy Control With Sarsa
- <img src="../images/on_policy_control_with_SARSA.png" alt="Agent and Environment" width="35%">
- <img src="../images/SARSA_action_value_update.png" alt="Agent and Environment" width="35%">
- <img src="../images/SARSA_on_policy_control_algo.png" alt="Agent and Environment" width="35%">
- <img src="../images/convergence_of_SARSA.png" alt="Agent and Environment" width="35%">
- <img src="../images/n_step_SARSA.png" alt="Agent and Environment" width="35%">

### Sarsa-Lambda, On-Policy
- <img src="../images/SARSA_lembda_forward_view.png" alt="Agent and Environment" width="35%">
- <img src="../images/SARSA_lembda_backward_view.png" alt="Agent and Environment" width="35%">
- <img src="../images/SARSA_lembda_algo.png" alt="Agent and Environment" width="35%">

### Off-Policy Learning
- <img src="../images/off_policy_learning.png" alt="Agent and Environment" width="35%">

### Importance Sampling
- Importance Sampling is useful for off-policy learning, to convert distribution from miu (ref policy) to pai (current policy)
- <img src="../images/importance_sampling.png" alt="Agent and Environment" width="35%">
- <img src="../images/importance_sampling_for_off_policy_MC.png" alt="Agent and Environment" width="35%">
- Off-Policy MC is not good due to very big variance. TD0 has much less variance since only look ahead for 1 step
- <img src="../images/importance_sampling_for_off_policy_TD.png" alt="Agent and Environment" width="35%">

### Q-Learning: Off-Policy Learning using Q(s, a)
- Importance sampling is NOT required for Q-Learning. Only V(s) needs importance sampling
- For Q-Learning, next action is chosen by behaviour (old) policy (miu), but Q(s,a) is updated using action chosen by target policy (pai)
- <img src="../images/Q_learning.png" alt="Agent and Environment" width="35%">
- <img src="../images/Q_learning1.png" alt="Agent and Environment" width="35%">
- <img src="../images/Q_learning_algo.png" alt="Agent and Environment" width="35%">
- <img src="../images/Q_learning_algo1.png" alt="Agent and Environment" width="35%">

### Relationship Between DP and TD
- DP do full-backup, TD do sample backup
- <img src="../images/relationship_btw_DP_and_TD.png" alt="Agent and Environment" width="35%">
- <img src="../images/relationship_btw_DP_and_TD1.png" alt="Agent and Environment" width="35%">


# L6 - Value Function Approximation

### Value Function Approximation
- Lookup table not suitable for large MDPs due to large number of states
- <img src="../images/value_function_approximation.png" alt="Agent and Environment" width="35%">
- <img src="../images/types_of_value_function_approximation.png" alt="Agent and Environment" width="35%">

### Differentiable Function Approximators
- <img src="../images/differentiable_function_approximators.png" alt="Agent and Environment" width="35%">


## Incremental Methods:

### Gradient Descent
- <img src="../images/gradient_descent.png" alt="Agent and Environment" width="35%">

### Value Function Approximation by Stochastic Gradient Descent
- <img src="../images/value_function_approximation_by_stochastic_gradient_descent.png" alt="Agent and Environment" width="35%">

### Feature Vectors
- <img src="../images/feature_vector.png" alt="Agent and Environment" width="35%">

### Linear Value Function Approximation
- <img src="../images/linear_value_function_approximation.png" alt="Agent and Environment" width="35%">

### Target Value Function for MC and TD
- <img src="../images/target_value_function_for_MC_and_TD.png" alt="Agent and Environment" width="35%">
- <img src="../images/linear_value_function_approximation_MC.png" alt="Agent and Environment" width="35%">
- <img src="../images/linear_value_function_approximation_TD0.png" alt="Agent and Environment" width="35%">
- <img src="../images/linear_value_function_approximation_TD_lambda.png" alt="Agent and Environment" width="35%">

### Control with Action-Value Function Approximation
- <img src="../images/control_with_value_function_approximation.png" alt="Agent and Environment" width="35%">
- <img src="../images/action_value_function_approximation.png" alt="Agent and Environment" width="35%">
- <img src="../images/linear_action_value_function_approximation.png" alt="Agent and Environment" width="35%">
- <img src="../images/linear_action_value_function_approximation_MC_and_TD0_and_TD_lambda.png" alt="Agent and Environment" width="35%">

### Convergence of Prediction Algorithms
- <img src="../images/convergence_of_prediction_algorithms.png" alt="Agent and Environment" width="35%">
- GTD is better in terms of convergence
- <img src="../images/gradient_TD_learning.png" alt="Agent and Environment" width="35%">
- <img src="../images/gradient_TD_learning1.png" alt="Agent and Environment" width="35%">

### Convergence of Control Algorithms
- <img src="../images/convergence_of_control_algorithms.png" alt="Agent and Environment" width="35%">


## Batch Methods:

### Batch Reinforcement Learning
- <img src="../images/batch_RL.png" alt="Agent and Environment" width="35%">
- Can be On-Policy or Off-Policy
- Experience Replay, Minimize Least Square Error of replay and value approximation
- <img src="../images/batch_RL_experience_replay.png" alt="Agent and Environment" width="35%">

### Batch RL using Stochastic Gradient Descent 
- <img src="../images/batch_RL_experience_replay_SGD.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_DQN.png" alt="Agent and Environment" width="35%">

### Batch RL using Linear Least Squares Prediction
- <img src="../images/batch_RL_experience_replay_linear_prediction.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_prediction1.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_prediction2.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_prediction3.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_prediction4.png" alt="Agent and Environment" width="35%">

### Batch RL using Least Squares Policy Iteration
- <img src="../images/batch_RL_experience_replay_least_square_policy_iteration.png" alt="Agent and Environment" width="35%">
- Linear Least Squares Action-Value Function Approximation
- <img src="../images/batch_RL_experience_replay_linear_least_square_action_value_func_approximation.png" alt="Agent and Environment" width="35%">

### Batch RL using Linear Least Squares Control
- <img src="../images/batch_RL_experience_replay_linear_least_square_control.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_least_square_control_Q_learning.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_least_square_control_LSTDQ.png" alt="Agent and Environment" width="35%">
- <img src="../images/batch_RL_experience_replay_linear_least_square_control_convergence.png" alt="Agent and Environment" width="35%">











