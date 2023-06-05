
import os
from unityagents import UnityEnvironment
import numpy as np
import torch
# from ddpg_agent import Agent, OUNoise, ReplayBuffer
from maddpg import MaddpgAgent

if __name__ in "__main__":
    root_dir = "C:/Users/johnb/repos/Udacity/deep-reinforcement-learning/p3_collab-compet/Tennis_Windows_x86_64"
    env = UnityEnvironment(file_name=root_dir+"/Tennis.exe")
    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    NUM_EPISODES = 1000
    MAX_STEPS = 1000
    # NOISE = 0.5
    NOISE_REDUCTION = 1.0
    STEPS_PER_UPDATE = 1

    num_episodes = NUM_EPISODES
    max_step = MAX_STEPS
    # noise = NOISE
    noise_reduction = NOISE_REDUCTION
    steps_per_update = STEPS_PER_UPDATE

    maddpg_agent = MaddpgAgent(state_size, action_size, discount_factor=0.99, tau = 1e-2,
                        num_agents=num_agents, buffer_size=int(1e5), batch_size=128, seed=44)
    
    for episode in range(int(1e3)):                                         # play game for 5 episodes
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        obs = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(num_agents)                          # initialize the score (for each agent)
        for step in range(max_step):
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            # print(actions)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_obs = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            score += rewards                                   # update the score (for each agent)
           
            if np.any(dones):                                  # exit loop if episode finished
                break

            maddpg_agent.add(obs, actions, rewards, next_obs, dones)

            obs = next_obs                              # roll over states to next time step
    print("Memory Full...")

    total_steps = 0
    scores = []
    best_score = 0
    for episode in range(num_episodes):                                         # play game for 5 episodes
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        obs = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            # actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            # actions = maddpg_agent.get_actions(obs, noise) # select an action (for each agent)
            actions = maddpg_agent.get_actions(obs) # select an action (for each agent)
            # actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            # print(actions)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_obs = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            maddpg_agent.update(obs, actions, rewards, next_obs, dones)
            score += rewards                                   # update the score (for each agent)
           

            obs = next_obs                              # roll over states to next time step

            # noise *= noise_reduction
            total_steps += 1

            if np.any(dones):                                  # exit loop if episode finished
                break

        scores.append(np.max(score))

        if (episode % 100 == 0 or episode == num_episodes-1) and episode != 0:
            # print(actions)
            # print(obs)
            mean_scores = np.mean(scores[-100:])
            print('Ep: {} Average score: {}'.format(episode, mean_scores))

            if mean_scores > best_score:
                #saving model
                save_dict_list =[] 
                for i in range(2):

                    save_dict = {'actor_params' : maddpg_agent.agents[i].actor_local.state_dict(),
                                'actor_optim_params': maddpg_agent.agents[i].actor_optimizer.state_dict(),
                                'critic_params' : maddpg_agent.agents[i].critic_local.state_dict(),
                                'critic_optim_params' : maddpg_agent.agents[i].critic_optimizer.state_dict()}
                    save_dict_list.append(save_dict)

                    torch.save(save_dict_list, 
                            os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
                best_score = mean_scores

    env.close()