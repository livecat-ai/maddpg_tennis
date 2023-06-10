# Maddpg Tennis
Part 3 of the Udacity Deep reinforcement learning course. An implementation of the maddgp reinforcement learning algorithm to solve the the Unity Tennis environment.



# The Environment - Introduction

## The Environment
This project, worked with the Tennis environment.

![Cooperative agents playing tennis](MaddpgPong.gif)
*Trained agents playing tennis*

Unity ML-Agents Tennis Environment
Unity ML-Agents Tennis Environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Installation



1. Install Conda on Windows.
Create a env Conda using Python 3.6
conda create --name drlnd python=3.6 
3. Activate this env. `conda activate drlnd`
4. Within this environment, you will need to install the necessary packages manually using PIP.

4.1 Install torch==0.4.0 manually: Download torch 0.4.0 wheel from http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl

Install Packages using PIP. First of all, you should go to your workspace, and download all your work using the following command in a cell from your notebook:
 

!tar chvfz notebook.tar.gz *
Then, unzip this file, and go to the directory where you unzipped it. In this directory, open a terminal and introduce the next command to move to the python folder.

cd python
Using the terminal in this folder, introduce the next command in order to install all the packages:

pip install .
In case of problems with some of the packages, just remove it from the requirements.txt file and try to install it manually. I seem to remember that the torch==0.4.0 package is missing. If this installation gives you your problem, there are two solutions:

1) Modify the torch==0.4.1 file, as this version is also supported.

2) Install torch==0.4.0 manually: Download torch 0.4.0 wheel from http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl. Once downloaded, introduce:

pip install --no-deps FILE-PATH\torch-0.4.0-cp36-cp36m-win_amd64.whl