import numpy as np
from ddpg_agent import Agent, OUNoise, ReplayBuffer

class MaddpgAgent:
    def __init__(self, obs_size, action_size, discount_factor, 
                 tau, num_agents, buffer_size, batch_size, seed):
        super(MaddpgAgent, self).__init__()

        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size= action_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.iter = 0

        self.agents = [Agent(self.obs_size, self.action_size, self.seed, 0.0)
                             for _ in range(num_agents)]
        
        
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed)

    def get_actions(self, obs, noise=True):
        # actions = [self.agents[i].act(obs[i]) for i in range(self.num_agents)]
        # return actions
        actions = []
        for ob, agent in zip(obs, self.agents):
            action = agent.act(ob, noise)
            actions.append(action)
        return actions

    def update(self, obs, actions, rewards, next_obs, dones):
        for i in range(self.num_agents):
            self.memory.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

        if self.memory.is_ready():
            for agent in self.agents:
                samples = self.memory.sample()
                agent.learn(samples, self.discount_factor)


    # def add(self, obs, actions, rewards, next_obs, dones):
    #     for i in range(self.num_agents):
    #         self.agents[i].memory.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

    # def update(self, obs, actions, rewards, next_obs, dones):
    #     for i in range(self.num_agents):
    #         self.agents[i].step(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

