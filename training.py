import logging

from network import PPO
from memory import PPOMemory

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


device = torch.device('cuda')

# def __init__(self, obs_h, obs_w, channels, actions_n):

def training(params, env):
    logger = logging.getLogger('obstacle_logger')
    T = params['time_horizon']
    episodes = params['episodes']
    learning_rate = params['lr']

    _obs = env.observation_space.shape

    memory = PPOMemory()
    # score = 0.0
    # print_interval = 20
    reward_sum=0
    reward_list = []
    timestep = 0
    # action_mask = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                0, 1, 1, 1]
    # n_actions = action_mask.count(1)
    action_mapping = {i:i*3 for i in range(18)}
    n_actions = len(action_mapping)

    model = PPO(_obs[0], _obs[1], _obs[2], n_actions).cuda()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)


    for n_epi in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            for t in range(T):
                # retro=False
                prob_action = model.actor(torch.tensor(state, dtype=torch.float, device=device), softmax_dim=1)
                # print(torch.sum(prob_action))
                # print(prob_action)
                
                # action = [Categorical(prob).sample().item() for prob in prob_action]
                # action_idx = np.array(action)
                action = Categorical(prob_action).sample().item()
                print(action)
                # print(action_mapping[action])
                # print(action)
                
                # print(action)
                # print(action_idx)
                
                # print(action_mapping[action_idx])
                
     
                next_state, reward, done, _ = env.step(action_mapping[action])

                memory.put_data((state, action, reward, next_state, prob_action[action].item(), done))
                state = next_state
                reward_sum += reward
                
                if done:
                    break
                
            timestep+=t
            optimizing(params, model, optimizer, memory)

#         if n_epi%print_interval==0 and n_epi!=0:
#             print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
#             score = 0.0

#         print(n_epi, reward_sum, timestep)
        logging_info = str(n_epi)+" "+str(reward_sum)+" "+str(timestep)
        logger.info(logging_info)
        torch.save(model.state_dict(), "n_epi"+str(n_epi))
        reward_list.append(reward_sum)
        reward_sum=0

    env.close()
    plt.plot([i for i in range(3000)], reward_list)
    # plt.show()
    plt.savefig('result.png')
    
    
def optimizing(params, model, optimizer, memory):
    logger = logging.getLogger('obstacle_logger')

    k_epoch = params['k_epoch']
    param_gamma = params['gamma']
    param_lambda = params['lambda']
    epsilon = params['epsilon']
    state, action, reward, next_state, done_mask, prob_action = memory.make_batch()

    for i in range(k_epoch):
        advantage_list = []
        advantage = 0.0
        state_value = model.critic(state)
        next_stateValue = model.critic(next_state)
        
        td_target = reward + param_gamma*next_stateValue*done_mask
        delta_list = td_target - state_value
        
        delta_list = delta_list.cpu().detach().numpy()
        for delta in delta_list[::-1]:
            advantage = param_gamma*param_lambda*advantage + delta[0]
            advantage_list.append([advantage])
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float, device=device)
        
        state_prob = model.actor(state, softmax_dim=1)
        # logger.info(str(state_prob.shape)+" "+str(action.shape))
        # logger.info(str(state_prob))
        # logger.info(str(action))
        # logger.info(str(state_prob.gather(1, action)))
        # # print(state_prob.shape, action.shape)
        # print(state_prob)
        # print(action)
        
        
        new_prob_action = state_prob.gather(1, action)
        
        ratio = torch.exp(torch.log(new_prob_action) - torch.log(prob_action))
        ra = ratio * advantage.detach()
        rclip = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage.detach()
        
        loss = -torch.min(ra, rclip) + F.smooth_l1_loss(state_value, td_target)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        