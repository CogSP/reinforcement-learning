# Tabular Solution Methods

# Multi-armed Bandits Problem

Only a single state

## k-armed bandit

each of the k action has a mean (expected) reward, also called the value $q_{\*}(a) = \mathbb{E}\[R_t | A_t = a\]$ of the function. So $R_t$ is the reward obtained when performing $A_t$, while $q_{\*}(a)$ is the expected reward.

If you knew the value $q_{\*}$ for each action the problem would be trivial, just choose the action with highest values $q_{\*}$.

We can estimate $q_{\*}$. We denote the estimated value of the action $a$ at time $t$ as $Q_t(a)$, and we would like to have $Q_t(a)$ as close as possible to $q_{\*}$

- Greedy algorithm: at time $t$ select the action $a$ with greatest estimate $Q_t(a)$ of the value. You are doing just exploitation.

We need to balance exploration and exploitation.

## Action-Value Methods

Estimating $Q_t(a)$ and using the estimate to make action selection decision. We can estimate $Q_t(a)$ as the average of the reward actually received (sample-average method):

$$
Q_t(a) = \frac{\text{sum of rewards when a was taken, before t}}{\text{number of times a was taken before t}} = \frac{\sum_{i=1}^{t-1} R_i * 1_{A_i = a}}{\sum_{i=1}^{t-1} 1_{A_i = a}}. 
$$

If denominator is zero $Q_t(a)$ is a default value. **By the law of large number**, if the denominator diverges, $Q_t(a)$ goes to $q_t(a)$.


How can we use now the estimation? Select the action with the highest estimated value (greedy): $A_t = argmax_{a}Q_t(a)$

We can once in a while, with small probability $\epsilon$, select randomly from all the action, with equal probability. In this way we are doing exploration. This is called $\epsilon$-greedy methods. In this case the probability of selecting the optimal action converges to greater than $1 - \epsilon$, so near certainty.


$\epsilon$-greedy performs better than greedy with noisier rewards, where we need more exploration to find the optimal action. This is large variance of the reward. On the other hand, if the reward variances were zero, the greedy method would konw the true value of each action after trying it once, so it would be better. However, we are assuming that the bandit task were stationary, so the true values of the actions don't change over time. If we assume nonstationary bandit problem, we need exploration.


## Incremental Implementation

Okay so we estimate the value with sample-average methods. And we have two methods, the greedy and the $\epsilon$-greedy. Now let's talk about how we can capmoute the averages to compute $Q_t(a)$ in a computationally efficient manner. Precisely with constant memory and constant per-time-step computation. 

Let's consider only one action. Let $R_i$ denote the reward received after the $i$-th selction of the action, and let $Q_n$ be the estimate of the action value after being selected $n-1$ times. So basically:

$$
Q_n = \frac{R_1 + R_2 + ... + R_{n-1}}{n - 1}
$$

If we would maintain a record of all the rewards and perform computation each time, the computational requirements and memory would grow over time as more rewards are seen. We can write an incremental formula for updating:

$$
Q_{n+1} = Q_n + \frac{1}{n}\[R_n - Q_n\]
$$

having manipulated the first equation. This requires to store only $Q_n$ and $n$ and from the computational point of view is a simple operation to perform. This formula is actually in the common form:

$$
\text{New Estimate} \longleftarrow \text{Old Estimate} + \text{Step Size}\[\text{Target} - \text{Old Estimate}\]
$$

You can see $\[\text{Target} - \text{Old Estimate}\]$ as an error in the estimate. Pseudo code for the $\epsilon$-greedy bandit problem
```
Initialize for each action
Q(a) <- 0
N(a) <- 0

Loop forever:
A <- random action with prob $\epsilon$, argmax_a Q(a) with probability $1 - \epsilon$
R <- bandit(A), basically apply A
N(a) <- N(A) + 1
Q(A) <- update rule above
```


## Nonstationary bandit problem

The update rule for $Q_{n+1}(a)$ discussed above is appropriate for stationary bandit problem, i.e. the reward probabilities don't change over time.

When we have a nonstationary problem, it makes sense to give more weight to recent rewards than long-past one. We can do that by giving constant step-size to the previous update rule.

$$
Q_{n+1} = Q_n + \alpha [R_n - Q_n]
$$

Indeed, with some manipulation, this will lead to

$$
Q_{n+1} = (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha * (1 - \alpha)^{n-i}R_i
$$

that is a weighted average of past rewards and the initial estimate $Q_1$ (the motivation for calling this weighted average are on the manual).

Note that, since $1 - \alpha$ is less than 1, the weight given to $R_i$ decreases as the number of rewards increases. So we actually gave more weight to recent rewards than long-past one. This is called exponential recency-weighted average. 


## Optimistic Initial Value: encouraging exploration with initial estimate

All the methods discussed so far are dependent on the initial action-value estimates $Q_1(a)$, so they are biased by their initial estimates. 

This means that the bias becomes a parameter that must be picked by the user. The upside is that they provide an easy way to supply some prior knowledge about what level of rewards can be expected. 


You can also use initial value to encourage exploration. For instance: in the k-th armed bandit we have initially set $Q(a) = 0$ for each action. What about setting them to +5? Recall that $q_{\*}(a)$ is selected from a normal distribution with mean 0 and variance 1. So initial estimate of +5 is wildly optimistic. This optimism encourages action-value methods to explore. Indeed, whichever action is chosen, its reward will be less than the starting estimate +5. So the learned will switch to other action, "disappointed" by the reward received. This means that all actions are tried several times. This holds also for greedy algorithm. 


Note that is is not well suited to nonstationary problems, because its drive for exploration is temporary (after convergence no more exploration). In nonstationary problem, the values change for we should renew the need for exploration, that is not possible with this method. 


## Upper-Confidence-Bound Action Selection (UCB)

$\epsilon$-greedy action selection forces the non-greedy action to be tried, but indiscriminately, with no preference for those that are nearly greedy or particularly uncertain. It would be better to select among the non-greedy actions **according to their potential for actually being optimal**, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates. You can do that with:

$$
A_t = argmax_{a}\[Q_t(a) + c\sqrt(\frac{\ln(t)}{N_t(a)})\]
$$

Note that $N_t(a)$ is the number of times the action $a$ was chosen prior to time $t$. $c > 0$ controls the exploration.

The square root term is a measure of the uncertainty or variance in the estimate of the action value. Each time $a$ is selected the uncertainty is presumably reduced: $N_t(a)$ increments and so the uncertainty term decreases. On the other hand when an action that is not $a$ is chosen, $t$ increases and $N_t(a)$ doesn't, increasing the uncertainty.



## Gradient Bandit Algorithms

The model learns a numerical preference $H_t(a)$ for each action $a$. The larger the preference the more often the action is taken. The learning algorithm is based on the idea of stochastic gradient ascent. 


# Contextual Bandits (Associative Search)

So far we have considered **nonassociative** tasks, so task in which there is no need to associate different actions with different situations. Either the model find the single best action (stationary) or tries to track the best action as it changes over time (nonstationary). However, in general there is more than one situation and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations. 

Example of associative search: your octopus facing the slot machine has now a clue on the identity of the task. The color of the display of the slot machine changes as the action values changes. Now you can learn an association between the state (color of the display) and the action (arm to pull). This association is called policy.

Associative search tasks are intermediate between the k-armed bandit problem (just 1 state) and the full reinforcement learning problem. Indeed, the involve a policy as the full RL problem, but each action affects only the immediate reward (as in the k-th armed bandit). On the contrary, if actions are allowed to affect the next situation as well as the reward, then we have the full RL problem. 


# Parameter study

<img src="images/parameter-study.png" alt="parameter study" width="393" height="219">


You can see the characteristic inverted-U shapes: the algorithms perform best at an intermediate value of their parameter.



# Bayesian bandits, Gaussian Bayesian bandits, Thompson Sampling

The Bayesian methods assume a known initial distribution over the action values and update the distribution exactly after each step, assuming the true action values $q_{\*}$ are stationary. At each step, we can select the action according to their posterior probability of being the best action. This is called posterior sampling or Thompson sampling. In the Bayesian settings it is even conceivable to compute the optimal balance between exploration and exploitation, often approximated efficiently since the computation can be immense.



# Finite Markov Decision Processes

MDP involves evaluative feedback as in bandits, but also an associative aspect: choosing different actions in different situations. MDP formalizes sequential decision making, where actions influence not just immediate reward, but also subsequent situations (states), through future rewards. MDP involves indeed the trade off between immediate and delayed rewards. While in bandits we estimate $q_{\*}(a)$ for each action, in MDPs we estimate $q_{\*}(s, a)$ for each action in each state, or we estimate the value $v_{\*}(s)$ of each state given optimal action selection.

## Agent-Environment Interface

The agent select actions and the environment responds to these actions and presents new situations (state) to the agent. The two interacts at each of a sequence of discrete time step $t = 0, 1, ...$. At each time step $t$ the agent receives some representation of the environment state $S_t \in S$ and on that basis select an action $A_t \in A(S)$. To simplify notation we often assume that the action set is the same in all states, writing $A$. One time step later (so at $t + 1$), the agent receives the reward $R_{t+1}$ and finds itself in a new state. So the sequence is this one:

$$
S_0 \ A_0 \ R_1 \ S_1 \ A_1 \ R_2 \ ...
$$

In a finite MDP $S$, $A$ and $R$ are finite and the random variables $R_t$ and $S_t$ have discrete probability distrbutino that depends only on the preceding state and action, so:

$$
p(s', r | s, a) = Pr\{S_t=s', R_t = r | S_{t-1} = s, A_{t-1} = a\}
$$

This function $p$ defines the environment dynamics in the MDP. Since the new state and reward depends only on the **immediately preceding** state and action, the state must include information about all aspects of all the past agent-environment interaction. If it does, the state is said to have the **Markov Property**. Other quantities that we may want to know:
- state transition probability: $p(s' | s, a) = Pr{S_t = s' | S_{t-1} = s, A_{t-1} = a}$
- expected rewards for state-action pairs: $r(s,a) = E[R_t | S_{t-1} = s, A_{t-1} = a]$
- expected rewards for state-action-next-state triples: $r(s,a,s') = E[R_t | S_{t-1} = s, A_{t-1} = a, S_t = s']


## Goals and Rewards

The goal of the agent is to maximixe not immediate reward, but the expected value of the cumulative sum of the reward in the long run. The reward signal is your way of communicating to the agent **what** you want achieved, not **how** you want it achieved. For the **how** is better ot impart a prior knowledge with an initial policy or an initial value function, as we have seen.


## Returns and Episodes

An episode is a subsequence of the whole agent-environment interaction sequence. For instance, if we are playing chess, a game is an episode. Each episode ends in a terminal state, followed by a reset to a starting state. Each episode is begins independently of how the previous one ended. If the task is episodic, you may want to return

$$
G_t = R_{t+1} + .. + R_{T}
$$

But what if the interaction doesn't break naturally in subsequences? Like an always on-going process-control task? In that case $T = \infty$, so the return $G_t$ it's infinite.  In this case we use the discounted return:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + … = R_{t+1} + \gamma G_{t+1}
$$

where $\gamma$ is the discount rate. 
- If $\gamma < 1$ we have that $G_t$ has a finite value as long as the reward sequence $\{R_k\}$ is bounded. 
- If $\gamma = 0$ the agent is myopic and maximizes the immediate reward. 
- As $\gamma$ goes to 1, the future rewards become stronger and the agent more farsighted 


## Policies and Value Functions

The value functions $v(s) estimate how good is for the agent to be in a certain state, or to perform a certain action in a certain state. The notion of "how good" is defined in terms of expected future rewards. Value functions use policies, so they are actually $v_{\pi}(s)$, and their definition is

$$
v_{\pi}(s) = E_{\pi}[G_t | S_{t} = s]
$$

A policy $\pi$ is a mapping from states to probabilities of selecting each possible action, so if an agent at time t follows the policy $\pi(a|s)$, it will calculate the probability that $A_t = a$ if $S_t = s$ for each a. 


### Bellman Equation for the value function given a policy

Playing with $v_{\pi} = E_{\pi}[G_t | S_{t} = s]$ we can obtain the Bellman Equation:

$$
v_{\pi} = sum_a \pi(a|s) \sum_{s', r} p(s',r | s,a)[r + \gamma v_{pi}(s') \ \ \forall s \in S
$$

This equation expresses a relationship between the value of a state $s$ and the values of its successor states. Think of looking ahead from state $s$ to its possible successor states: starting from the root node $s$, the agent could take any action based on $\pi$, and from each of these actions the env could respond with a state $s'$ along with a reward $r$, depending on its dynamics $p$. The Bellman Equation averages over all the possibilities, weighting each possibility by its probability of occurring.


## Optimal Policies and Optimal Value Functions

Solving an RL task means finding the optimal policy. A policy $\pi$ is better than $\pi'$ if its expected return is greater than that of $\pi'$ for all states. In other words

$$
\pi \geq \pi' \Leftrightarrow v_{\pi}(s) \geq v_{\pi'}(s)
$$

The optimal policy (or policies) is (are) $\pi_{*}$.  The optimal state-value function is 

$$
v_{*}(s) = max_{\pi} v_{\pi}(s) \ \forall s \in S
$$

and the optimal action-value function is:

$$
q_{*}(s, a) = max_{\pi} q_{\pi}(s, a) \ \forall s \in S, a \in A
$$

From these, we can write the Bellman Equation for the optimal action-value function

$$
q_{\*}(s,a) = E[R_{t+1} + \gamma v_{*}(S_{t+1}) | S_t = s, A_t = a] = \sum_{s', r}p(s', r | s,a)[r + \gamma max_{a'} q_{*}(s', a')]
$$


## Optimality and Approximation

In practice an agent rarely learns an optimal policy, since the computational power and memory needed may be preposterous. For small, finite state sets, we can use arrays or tables with one entry for each state-action pair: these are the tabular methods. In many case however, there are too much states and it's impossible to use a table. In these cases the functions must be approximated. One key property is that we can approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states. In other words, since we can't, for computational reason, do everything good, we do important things (since they are common) well, and neglectable things (the ones that are uncommon) badly.



# Dynamic Programming

DP are a collections of algo used to compute optimal policies given a perfect model of the env as a MDP. They assume perfect model and they are computational expense, but they are important whatsoever. DP algo turns the two Bellman Equations into update rules for improving the approximations of the desired value functions.

## Policy Evaluation (Prediction)

How to compute the state-value function $v_{\pi}$ for an arbitrary $\pi$? With a policy evaluation or prediction problem. Given the Bellman equation for $v_{\pi}$:+

$$
v_{\pi} = sum_a \pi(a|s) \sum_{s', r} p(s',r | s,a)[r + \gamma v_{pi}(s') \ \ \forall s \in S
$$

we can compute the solution, but it's computationally tedious. So we can use iterative solution methods: consider a sequence of approximate value functions $v_0, v_1, v_2, ...$, having chosen the initial approximation $v_0$ arbitrarily, and having computed every successive ones using the Bellman equation for $v_{\pi}$ but in the form of an update rule:

$$
v_{k+1} = sum_a \pi(a|s) \sum_{s', r} p(s',r | s,a)[r + \gamma v_{k}(s') \ \ \forall s \in S
$$

The sequence $\{v_k\} converges to $v_{\pi}$ as $k \to \infty$. This is the iterative policy evaluation. 

To implement it in a computer, you would have to use two arrays: one for the old values $v_k(s)$ and one for the new values $v_{k+1}(s)$, or update the values "in place". The in-place algo is chosen here:


Input: $\pi$, the policy to be evaluated
Algorithm parameter: a small threshold $\theta > 0$ determining accuracy of estimation 
Initialize $V(s)$ arbitrarily, for $s \in S$ and $V(\text{terminal}) = 0$ 

**Loop:**
$\ \ \ \Delta \leftarrow 0$ 
$\ \ \ \text{Loop for each } s \in S$: 
$\ \ \ \ \ \  \ v \leftarrow V(s)$ 
$\ \ \ \ \ \  V(s) \leftarrow \sum_{s'} \pi(a | s) \sum_{s', r}p(s', r | s,a)[r + \gamma V(s')]$
$\ \ \ \ \ \ \Delta \leftarrow max(\Delta, |v - V(s)|)$
$until\  \Delta < \theta$

Formally, the iterative policy evaluation converges only in the limit, but in practice it must be halted short of this, using the threshold $\theta$, as shown in the algorithm above.


## Policy Improvement

We have see how to compute the value function. We want to do that to find a better policy. Suppose we have found $v_{\pi}$ for an arbitrary $\pi$. We know, in a certain state s, how good it is to follow the current policy from s (it's $v_{\pi}(s)), but would it be better or worse to change to the new policy? One way to answer this question is to consider selecting a in s and thereafter following the existing policy $\pi$. So we can compute $q_{\pi}(s,a)$ with the Bellman Equation. Now, is this $q_{\pi}(s,a)$ greater than $v_{\pi}(s)$? If it is greater, it's better to select $a$ once in $s$ and thereafter to follow $\pi$ than to follow $\pi$ all the time. If this is true, one would expect it to be better still to select $a$ every time $s$ is encountered, and that the new policy, that select $a$ in $s$, would in fact be a better one overall. This expectation is proven by the **policy improvement theorem**: let $\pi$ and $\pi'$ be policies s.t. $\forall s \in S$:

$$
q_{\pi}(s, \pi'(s)) \geq v_{\pi}(s)
$$

Then $\pi'$ must be as good as, or better than $\pi$, so:

$$
v_{\pi'}(s) \geq v_{\pi}(s)
$$

So if $\pi'$ is equal to $\pi$ except for the action $a$ taken in $s$ (i.e. $\pi'(s) = a \neq \pi(s)), $\pi'$ should be used instead of $\pi$, since it's a better policy. It is a natural extension to consider changes between $\pi$ and $\pi'$ at all states, selecting at each state the action that appears best according to $q_{\pi}(s,a):

$$
\pi'(s) = argmax_{a} q_{\pi}(s,a) = … = argmax_{a} \sum_{s',r} p(s', r | s, a)[r + \gamma v_{\pi}(s')]
$$

This policy is called **greedy policy** and basically takes the action that looks best in the short term, according to $v_{\pi}$. This greedy policy will be as good as, or better than, the original policy. We are doing **policy improvement**. It can be shown that if $\pi'$ is as good as $\pi$, we have that $v_{\pi'}$ is $v_{\*}$ and so $\pi'$ and $\pi$ are both optimal policies. **Policy improvement thus must give us a strictly better policy except when the original policy is already optimal.**


### Stochastic Policies 

A stochastic policy $\pi$ specifies probabilities $\pi(a|s)$ for taking each action $a$ in each state $s$. All the ideas written above for the deterministic policies can be easily extended to the stochastic ones.


## Policy Iteration

Once a policy $\pi$ has been improved using $v_{\pi}$ to yield a better policy $\pi'$, we can compute $v_{\pi'}$ and improve it again to yield an even better $\pi''$, so we can obtain a sequence of **monotonically improving policies and value functions**:

$$
\pi_0 \to_{E} v_{\pi_0} \to{I} \pi_1 \to_{E} v_{\pi_1} \to{I} \pi_2 … \to_{I} \pi_{\*} \_to_{E} v_{\*}
$$
Where E denotes policy evaluation and I policy improvement. As we said, each policy is guaranteed to be a strict improvement over the previous one (unless it is already optimal). Since a finite MDP has only a finite number of deterministic policies, this process must converge to an optimal policy and its optimal value function in a finite number of iterations. This is policy iteration. The pseudocode is the following:


1. Inizialization: 
	$V(s) \in \mathbb{R}$ 
	$\pi(s) \in A(s)$ arbitrarily $\forall s \in S$
	V(terminal) = 0

2. Policy Evaluation
	**Loop:**
$\ \ \ \Delta \leftarrow 0$ 
$\ \ \ \text{Loop for each } s \in S$: 
$\ \ \ \ \ \  \ v \leftarrow V(s)$ 
$\ \ \ \ \ \  V(s) \leftarrow \sum_{s'} \pi(a | s) \sum_{s', r}p(s', r | s,a)[r + \gamma V(s')]$
$\ \ \ \ \ \ \Delta \leftarrow max(\Delta, |v - V(s)|)$
$until\  \Delta < \theta$

3. Policy Improvement
	$\text{policy-stable} \leftarrow true$
	For each $s \in S$
	$\ \ \ \text{old-action} \leftarrow \pi(s)$ 
$\ \ \ \pi(s) \leftarrow argmax_{a}\sum_{s',r}p(s',r |s,a)[r + \gamma V(s')]$	 
$\ \ \ \text{If old-action} \neq \pi(s), \text{then policy-stable} \leftarrow false$

If policy-stable, then stop and return$V \approx v_{*}$ and $\pi \approx \pi_{*}$; else go to 2
	
	
## Value Iteration

One drawback to policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation requiring multiple sweeps through the state set. We can truncate the policy iteration in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep, namely one update of each state. In this case, we talk about value iteration. The value iteration algorithm can be written with just a simple update operation combining the policy improvement and the truncated policy evaluation in just one formula:

$$
v_{k+1}(s) = max_{a} \sum_{s',r} p(s', r | s,a) [r + \gamma v_{k}(s')]
$$
To be performed for all $s \in S$. For an arbitrary $v_0$, the sequence $\{v_k}$ converges to $v_{*}$. Here's the pseudo-code:


Algorithm parameter: a small threshold $\theta > 0$ determining accuracy of estimation
Initialize V(s), $\forall s \in S^{+}$, arbitrarily except that V(terminal) = 0

**Loop**
$\ \ \ \Delta \leftarrow 0$
$\ \ \ \ \ \  v \leftarrow V(s)$
$\ \ \ \ \ \ V(s) \leftarrow max_{a} \sum_{s',r} p(s', r |s,a)[ r + \gamma V(s')]$
$\ \ \ \ \ \ \Delta \leftarrow max(\Delta, |v - V(s)|)$
until $\Delta < \theta$

Output a deterministic policy, $\pi \approx \pi_{*}$, s.t. 
$\ \ \ \pi(s) = argmax_{a} \sum_{s',r} p(s', r | s, a)[r + \gamma V(s')]$


As you can see we are combining one sweep of policy evaluation and one sweep of policy improvement in each of its sweeps.
