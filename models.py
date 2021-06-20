# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:46:20 2021

@author: 6794c
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def epsilon_greedy_batch(q_function, epsilon, n_actions):
    if random.random() > epsilon:
        return torch.argmax(q_function, dim=1)
    else:
        return torch.randint(0, n_actions, (len(q_function), ))


def epsilon_greedy(q_function, epsilon, n_actions):
    if random.random() > epsilon: # greedy
        return np.argmax(q_function.cpu().detach().numpy())
    else:
        return random.randint(0,n_actions-1)
    

class PPO(nn.Module):
    def __init__(self, obs_h, obs_w, channels):
        super(PPO, self).__init__()
        self.data = []
        # self.observation = (obs_h, obs_w, channels)
        self.observation = (obs_h, obs_w, 12)
        
        # self.input = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=4)
        self.input = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        
        # pytorch dqn
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(obs_h, 8, 2), 4, 2), 3, 1)
        
        self.fc_actor = nn.Linear(in_features= 7*7*64, out_features=512)
        self.output_actor = nn.Linear(in_features=512, out_features=12)
        # env.action_space MultiDiscrete([3 3 2 3])
        # forward/backward/no-op
        # left/right/no-op
        # jump/no-op
        # clockwise/counter-clockwise rotation camera/no-op
        # self.
        
        self.fc_critic = nn.Linear(in_features = 7*7*64, out_features=512)
        self.output_critic = nn.Linear(in_features=512, out_features=1)

        
    def actor(self, x, softmax_dim = 0):
        # print(x.shape)
        
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        # print(x.shape)
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc_actor(x))
        x = self.output_actor(x)

        prob = F.softmax(x, dim=softmax_dim)

        return prob
    
    def critic(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])

        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc_critic(x))
        value = self.output_critic(x)

        return value


class MetaQNetController(nn.Module):
    def __init__(self, obs_h, obs_w, channels):
        super(MetaQNetController, self).__init__()
        self.observation = (obs_h, obs_w, 12)
        
        self.input = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        
        # pytorch dqn
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(obs_h, 8, 2), 4, 2), 3, 1)
        
        self.fc = nn.Linear(in_features= 7*7*64, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=2)
        
        
        nn.init.kaiming_normal_(self.input.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')

        
    def forward(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc(x))
        x = self.output(x)

        return x
    
    
class ObsQNetController(nn.Module):
    def __init__(self, obs_h, obs_w, channels):
        super(ObsQNetController, self).__init__()
        self.observation = (obs_h, obs_w, 12)
        
        self.input = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        
        # pytorch dqn
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(obs_h, 8, 2), 4, 2), 3, 1)
        
        self.fc = nn.Linear(in_features= 7*7*64, out_features=512)
        # force observation = (0, 1, 1, 0), (0, 2, 1, 0) / 9, 15
        self.obs_output = nn.Linear(in_features=512, out_features=2)
        
        self.action_output = nn.Linear(in_features=512, out_features=36)
                
        nn.init.kaiming_normal_(self.input.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')


    def obs_forward(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        # x = torch.cat((x, meta_action))
        
        x = F.relu(self.fc(x))
        x = self.obs_output(x)

        return x
    
    
    def action_forward(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        # x = torch.cat((x, meta_action))
        
        x = F.relu(self.fc(x))
        x = self.action_output(x)

        return x
    
    
###################################################################################
class ObservationOnlyQNet(nn.Module):
    def __init__(self, obs_h, obs_w, channels):
        super(ObservationOnlyQNet, self).__init__()
        self.observation = (obs_h, obs_w, 12)
        
        self.input = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        
        # pytorch dqn
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(obs_h, 8, 2), 4, 2), 3, 1)
        
        self.fc = nn.Linear(in_features= 7*7*64, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=2)

        
    def forward(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc(x))
        x = self.output(x)
        
        # if x==0: # left observation with jump
        #     return 9
        # else:
        #     return 15 # right observation with jump

        return x
    
class ActionOnlyQNet(nn.Module):
    def __init__(self, obs_h, obs_w, channels):
        super(ActionOnlyQNet, self).__init__()
        self.observation = (obs_h, obs_w, 12)
        
        self.input = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        
        # pytorch dqn
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(obs_h, 8, 2), 4, 2), 3, 1)
        
        self.fc = nn.Linear(in_features= 7*7*64, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=36)

        
    def forward(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc(x))
        x = self.output(x)


        # if x>=14:
        #     return x+2
        # if x>=9:
        #     return x+1
        # else:
        #     return x

        return x
    
    
"""
(1, 0, 0, 0) = forward  
(0, 0, 0, 0) = no-op  
(2, 0, 0, 0) = backward  
(0, 1, 0, 0) = 반시계방향 카메라  
(0, 2, 0, 0) = 시계방향 카메라  
(0, 0, 1, 0) = 점프  
(0, 0, 0, 1) = 우측으로 이동카메라 고정  
(0, 0, 0, 2) = 좌측으로 이동, 카메라  고정

action mapping
{0: [0, 0, 0, 0], 
 1: [0, 0, 0, 1],
 2: [0, 0, 0, 2],
 3: [0, 0, 1, 0],
 4: [0, 0, 1, 1],
 5: [0, 0, 1, 2],
 6: [0, 1, 0, 0],
 7: [0, 1, 0, 1],
 8: [0, 1, 0, 2],
 9: [0, 1, 1, 0], <- meta operation
 10: [0, 1, 1, 1],  9
 11: [0, 1, 1, 2],  10
 12: [0, 2, 0, 0],  11
 13: [0, 2, 0, 1],  12
 14: [0, 2, 0, 2],  13
 15: [0, 2, 1, 0],  <- meta operation
 16: [0, 2, 1, 1],  14
 17: [0, 2, 1, 2],  15
 18: [1, 0, 0, 0],  16
 19: [1, 0, 0, 1],
 20: [1, 0, 0, 2],
 21: [1, 0, 1, 0],
 22: [1, 0, 1, 1],
 23: [1, 0, 1, 2],
 24: [1, 1, 0, 0],
 25: [1, 1, 0, 1],
 26: [1, 1, 0, 2],
 27: [1, 1, 1, 0],
 28: [1, 1, 1, 1],
 29: [1, 1, 1, 2],
 30: [1, 2, 0, 0],
 31: [1, 2, 0, 1],
 32: [1, 2, 0, 2],
 33: [1, 2, 1, 0],
 34: [1, 2, 1, 1],
 35: [1, 2, 1, 2], 

 no backward
 36: [2, 0, 0, 0],
 37: [2, 0, 0, 1],
 38: [2, 0, 0, 2],
 39: [2, 0, 1, 0],
 40: [2, 0, 1, 1],
 41: [2, 0, 1, 2],
 42: [2, 1, 0, 0],
 43: [2, 1, 0, 1],
 44: [2, 1, 0, 2],
 45: [2, 1, 1, 0],
 46: [2, 1, 1, 1],
 47: [2, 1, 1, 2],
 48: [2, 2, 0, 0],
 49: [2, 2, 0, 1],
 50: [2, 2, 0, 2],
 51: [2, 2, 1, 0],
 52: [2, 2, 1, 1],
 53: [2, 2, 1, 2]
 """