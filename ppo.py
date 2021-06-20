# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:39:15 2021

@author: 6794c
"""
import os
import time
import logging

import numpy as np
import torch

from models import PPO

class PPOTrainer:
    def __init__(self, config):
        if not os.path.exists("summaries"):
            os.makedirs("summaries")
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
            
        # basic configuration setting
        self.logger_epi = logging.getLogger('obstacle_logger_epi')
        self.logger_default = logging.getLogger('obstacle_logger_default')
        self.env_config = config["environment"]
        self.training_config = config['training']
        self.default_config = config['default']
        self.worker_id = self.default_config['worker_id']
        self.n_workers = self.default_config['n_workers']
        self.model_config = self.training_config['model']
        timestamp = time.strftime("/%Y%m%d-%H%M%S"+ "_" + str(self.worker_id) + "/")        
        
        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # trainin configuration
        self.history_length = self.training_config['history_length']
        self.channel_size = self.history_length*3


    def sample(self):
        with torch.no_grad():
            


    def training(self):
        # model cpu, gpu setting
        if torch.cuda.is_available():
            self.meta_Qlearner.cuda()
            self.target_meta_Qlearner.cuda()
            self.obs_Qlearner.cuda()
            self.target_obs_Qlearner.cuda()
            self.action_Qlearner.cuda()
            self.target_action_Qlearner.cuda()

        # multiple_obs_actions = []
        # multiple_action_actions = []
        dones = np.zeros((self.n_workers), dtype=np.bool)
        states_list = []
        
        state = self.obs.transpose(2, 0, 1)
        
        # history initialization
        self.multiple_history = np.zeros((self.n_workers,)+(self.channel_size, 84, 84), dtype=np.float32)
        self.multiple_next_history = np.zeros((self.n_workers,)+(self.channel_size, 84, 84), dtype=np.float32)
        for i in range(len(self.workers)):
            self.multiple_history[i][0:3] = state[:]
        for i in range(0, self.channel_size, 3):
            for worker in self.workers:
                worker.parent_conn.send(("step", 0))
            for idx, worker in enumerate(self.workers):
                next_state, _, _, _ = worker.parent_conn.recv()
                next_state = next_state.transpose(2, 0, 1)
                history_idx = int((i+3)%self.channel_size)
                self.multiple_history[idx][history_idx:history_idx+3] = next_state[:]
                self.multiple_next_history[idx][i:i+3] = next_state[:]

        # update iteration
        for idx in range(self.updates):
            step_result = self.one_step()

            if self.meta_memory.size() >= 40000:
                
                
                self.optimizing()
            
            # if idx % self.model_checkpoint==0:
            #     torch.save(self.model, self.model_checkpoint_path+"-"+str(idx)+".pt")
            if idx%100==0:
                self.logger_epi.info(f'rewards : {step_result[0]:.2f}  floor : {step_result[1]:.2f}  timestamp : {step_result[2]:.2f}')
                # print(f'rewards : {step_result[0]:.2f}  floor : {step_result[1]:.2f}  timestamp : {step_result[2]:.2f}')