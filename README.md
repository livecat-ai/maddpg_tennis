# Maddpg Tennis
Part 3 of the Udacity Deep reinforcement learning course. An implementation of the maddgp reinforcement learning algorithm to solve the the Unity Tennis environment.



# The Environment - Introduction

## The Environment
This project, worked with the Tennis environment.

![Cooperative agents playing tennis](MaddpgPong.gif)
*Trained agents playing tennis*

### Unity ML-Agents Tennis Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Installation

## Important
Because the exercises in this course haven't been updated for quite some time, many of the requirements needed to run the Unity Tennis enviroment are depriciated and no longer work. Because of this, I was unable to get the environment running under current version of Linux or OSX.

I could however finally get the code running under Windows10 and Windows11. It maybe be possible that the environment would run on older Linux versions but I didn't try that.

### Setup Conda on Windows
1. Install Conda on Windows.
Create a Conda env using Python 3.6
`conda create --name drlnd python=3.6`
2. Activate this env. `conda activate drlnd`
3. Within this environment, you will need to install the necessary packages manually using PIP.
Install Packages using PIP.

### Download and install torch 0.4.0
Torch 0.4.0 is no longer supported. Follow the step below to install it.

1. Install torch==0.4.0 manually: Download torch 0.4.0 wheel from http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl
2. `pip install --no-deps FILE-PATH\torch-0.4.0-cp36-cp36m-win_amd64.whl`

### Clone the deep-reinforcement-learning repository from Github
1. Clone the Udacity Deep Reinforcement Learning repository: `git clone https://github.com/udacity/deep-reinforcement-learning.git`

### Install the required package using pip
1. `cd deep-reinforcement-learning/python`
2. Type: `pip install .` to install all the required packages

### Download and install the Unity Tennis environment
1. Then follow the instruction here to download the and install the Unity environment: https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731
2. Be sure to install the Windows version supported by your machine.

### Download this repository
1. Download this repository to the p3_collab-compet/ folder.

# Running the code
1. cd to /deep-reinforcement-learning/p3_collab-compet/maddpg_tennis/. This directory contains all the code needed to train the agents.
2. run `jupyter notebook` and open tennis.ipynb. Then run the notebook. Training will start. When finished a graph will be display at the bottom of the notebook showing the learning values for the maddpg agent. You can than close the notebook.
3. To see the agent play tennis. open Play.ipynb and run the notebook. A Unity window will open and show the agents playing tennis, much like the one at the top of this page.

# Report
I tried many experiments attempting to implement the full MADDPG algorithm as described in the OpenAI paper. However I could get it to work on this environment.

In the end I adapted the code for the ddpg-pendulum environment provided in the deep reinforcement learning repository. The Code was adapted by adding a Maddpg class to manage two independent ddpg agents and a shared replay buffer.

The Actor and Critic models were both two layer MLP with 128 node each. 
The parameters used were:
`
NUM_EPISODES = 1000
MAX_STEPS = 1000
STEPS_PER_UPDATE = 1
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
`

Learning was vastly improved by adding dropout layers to both networks. Oddly this was the case even if the dropout rate was set to zero. I'm not sure why this would be, so it needs looking into futher.

## Results
Then agent reaches the learning goal of 0.5 after about 650 episode. And reaches a maximum score of around 1.4 within a 1000 episodes. 

![Alt text](results.png)


## Futher experiments
A couple of things to try which could improve learning.

1. Reducing the process noise over time might improve improve the highest score and stabilize learning.

2. Using parameter noise instead of adding noise to the action values. This seem very promising. https://openai.com/research/better-exploration-with-parameter-noise

3. Implement a different underlying algorithm like SAC or PPO.

4. Spending more effort on parameter tuning or play with the model architectures.






