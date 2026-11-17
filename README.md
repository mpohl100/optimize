# Motivation

This GitHub repository is a collection of my Rust implementations of my favorite optimization algortihms in Rust.
It contains:
  - a library for linear regression on the CPU with efforts to target the CPU
  - a library for genetic algorithms
  - a library for counterfactual regret minimization
  - a library to optimize Markov processes using counterfactual regret minimization

These crates are not meant for production use, they are meant to enable my research that I find intereseting for AI.

# Goal

Linear regression is the core algortihm for modern AI.
My goal is to implement a program which computes the optimal neural network architecture for a given dataset.
My idea on how to achieve this is to improve linear regression runs by switching off decisive weights during training so that other regions of the neural network have to carry more of the load to compute the correct prediction. However it is difficult to decide which weights to switch off at a certain point in time.
The gist is to use counterfactual regret minimization for making the decision which weights to switch off.
The decision which parts of the neural network to keep inactive is essentially a game of incomplete information - at the time of the decision we don't know whether the decision will work out.
As counterfactual regret minimization is the algorithm used in Poker solvers to compute the optimal percentages on which decision to take and it essentially turned the game of Poker into a solved game, I see high potential in leveraging this algorithm and to combine it with linear regression.

# Current state

[x] linear regression libary on the CPU
[x] generic counterfactual regret solver
[] first POC to combine linear regression with counterfactual regret minimization

# Outlook

If the idea turns out to be useful I expect the possibility of significantly reducing AI training costs.

