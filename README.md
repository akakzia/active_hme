# Help Me Explore: Minimal Social Interventions for Graph-Based Autotelic Agents
_This repository contains the code associated to the "Help Me Explore: combining autotelic and social learning via active goal queries" paper submitted at the ICLR 2023 conference._

**Abstract.**
Most approaches to open-ended skill learning train a single agent in a purely sensorimotor world. But because no human child learns everything on their own, we argue that sociality will be a key component of open-ended learning systems. This paper enables learning agents to blend individual and socially-guided skill learning through a new interaction protocol named Help Me Explore (HME).
In social episodes triggered at the agent's demand, a social partner suggests a goal at the frontier of the agent's capabilities and, when the goal is reached, follows up with a new adjacent goal just beyond.
In individual episodes, the agent practices skills autonomously by pursuing goals it has already discovered through either its own experience or social suggestions.
The idea of augmenting an individual goal exploration with social goal suggestions is simple, general and powerful. We demonstrate its efficiency on hard exploration problems: continuous mazes and a 5-block robotic manipulation task. With minimal social interventions, the \hme agent outperforms both the purely social and purely individual agents.

**Link to Website**

Link to our website will be available soon with additional illustrations and videos.

**Requirements**

* gym
* mujoco
* pytorch
* pandas
* matplotlib
* numpy
* networkit

To reproduce the results, you need a machine with **20** cpus.

**Simulated Social Partner**

The _HME_ interaction protocol relies on a simulated social partner that possesses a model of the agent's learned and learnable skills. 
We represent this knowledge with a semantic graph of connected configurations. This graph is already generated and available under 
the _graph/_ folder.

**Training GANGSTR**

The following line trains the GANGSTR agent with social intervention ratio of 20% and a Frontier and Beyond Strategy

```mpirun -np 20 python train.py --env-name FetchManipulate5Objects-v0 --agent --beta 50 'HME' ```

When beta is equal to 0, the GANGSTR agent only performs social learning. By contrast, when beta is higher than 200, only individual episodes are conducted. 
