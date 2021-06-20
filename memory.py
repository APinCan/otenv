# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:04:17 2021

@author: 6794c
"""
import random
from collections import deque

import torch
import numpy as np

device = torch.device('cuda')


class PPOMemory:
    def __init__(self):
        self.data = []
                
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        entropy_list = []
        
        for transition in self.data:
            # s, a, r, s_prime, prob_a, done = transition
            s, a, r, s_prime, prob_a, entropy, done = transition

            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            # entropy_list.append([entropy])
            
        # s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float, device=device), torch.tensor(a_lst, dtype=torch.int64, device=device), \
        #                                   torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, dtype=torch.float, device=device), \
        #                                   torch.tensor(done_lst, dtype=torch.float, device=device), torch.tensor(prob_a_lst, device=device)
                                          
        s,a,r,s_prime,done_mask, prob_a, entropy = torch.tensor(s_lst, dtype=torch.float, device=device), torch.tensor(a_lst, dtype=torch.int64, device=device), \
                                  torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, dtype=torch.float, device=device), \
                                  torch.tensor(done_lst, dtype=torch.float, device=device), torch.tensor(prob_a_lst, device=device), \
                                  torch.tensor(entropy_list, dtype=torch.float, device=device)
                                          
        self.data = []
        # return s, a, r, s_prime, done_mask, prob_a
        return s, a, r, s_prime, done_mask, prob_a, entropy


class Buffer:
    def __init__(self, device):
        self.vis_obs = np.zeros((4, 512)+(12, 84, 84), dtype=np.float32)
        self.values = np.zeros((4, 512), dtype=np.float32)
        self.actions = np.zeros((4, 512), dtype=np.int32)
        self.log_probs = np.zeros((4,512), dtype=np.float32)
        self.rewards = np.zeros((4, 512), dtype=np.float32)
        self.dones = np.zeros((4, 512), dtype=np.bool)
        self.advantages = np.zeros((4, 512), dtype=np.float32)
        self.minibatch_device = device
        
        
    def calc_advantage(self, last_value, gamma, lamda):
        last_advantage = 0
        for t in reversed(range(512)):
            mask = 1.0 - self.dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = self.rewards[:, t] + gamma * last_value - self.values[:, t]
            last_advantage = delta + gamma * lamda * last_advantage
            self.advantages[:, t] = last_advantage
            last_value = self.values[:, t]
            
    def prepare_batch_dict(self):
        samples = {
            'actions': self.actions,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'dones': self.dones
        }
        samples['vis_obs'] = self.vis_obs
        
        
        self.samples_flat = {}
        for key, value in samples.items():
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = torch.tensor(value, dtype=torch.float32, device=self.minibatch_device)
    
    
    def mini_batch_generator(self):
        # Prepare indices (shuffle)
        batch_size = 4 * 512
        mini_batch_size = batch_size // 4
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            # Arrange mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                mini_batch[key] = value[mini_batch_indices].to(self.minibatch_device)
            yield mini_batch
    
    

class ReplayPPOMemory:
    def __init__(self):
        self.data = []
        self.batch_index = 0
        self.size = 0
        
    def put(self, transition):
        self.data.append(transition)
        self.size += 1
        
    def endOfMemory(self):
        if self.batch_index >= self.size:
            return True
        else:
            return False
        
    def clean(self):
        self.data = []
        self.batch_index = 0
        self.size = 0
        
    def make_batch(self):
        state_list, action_list, reward_list = [], [], []
        nextState_list, probA_list, entropy_list, done_list = [], [], [], []
        T = self.batch_index+20
        
        for _ in range(self.batch_index, self.size):
            transition = self.data[self.batch_index]
            state, action, reward, next_state, prob_a, entropy, done = transition

            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            nextState_list.append(next_state)
            probA_list.append([prob_a])
            entropy_list.append([entropy])
            done_list.append([0 if done else 1])
        
            self.batch_index+=1
            if self.batch_index>=T:
                break

        state = torch.tensor(state_list, dtype=torch.float, device=device)
        action = torch.tensor(action_list, dtype=torch.int64, device=device)
        reward = torch.tensor(reward_list, device=device)
        next_state = torch.tensor(nextState_list, dtype=torch.float, device=device)
        prob_a = torch.tensor(probA_list, device=device)
        entropy = torch.tensor(entropy_list, dtype=torch.float, device=device)
        done = torch.tensor(done_list, dtype=torch.float, device=device)
        
        return state, action, reward, next_state, prob_a, entropy, done


class ReplayMemory():
    def __init__(self, memory_size):
        self.data = deque(maxlen=memory_size)
        self.max_size = memory_size
        self.capacity=0
        
    def put(self, transition):
        self.data.append(transition)
        if self.capacity < self.max_size:
            self.capacity+=1
        
    def size(self):
        return self.capacity
    

    def uniform_sample(self, batch_idx):
        # mini_batch = self.data[batch_idx]
        mini_batch = [self.data[i] for i in batch_idx]
        # print(mini_batch)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float, device=device), torch.tensor(a_lst, dtype=torch.int64, device=device), \
               torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, dtype=torch.float, device=device), \
               torch.tensor(done_mask_lst, device=device)
        
    # def sample(self, batch_size):
    #     mini_batch = random.sample(self.data, batch_size)
    #     s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
    #     for transition in mini_batch:
    #         s, a, r, s_prime, done_mask = transition
    #         s_lst.append(s)
    #         a_lst.append([a])
    #         r_lst.append([r])
    #         s_prime_lst.append(s_prime)
    #         done_mask_lst.append([done_mask])

    #     return torch.tensor(s_lst, dtype=torch.float, device=device), torch.tensor(a_lst, device=device), \
    #             torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, dtype=torch.float, device=device), \
    #             torch.tensor(done_mask_lst, device=device)
               
    def sample(self, batch_size):
        mini_batch = random.sample(self.data, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float, device=device), torch.tensor(a_lst, dtype=torch.int64, device=device), \
                torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, dtype=torch.float, device=device), \
                torch.tensor(done_mask_lst, device=device)