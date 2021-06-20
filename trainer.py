# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:22:06 2021

@author: 6794c
"""
import os
import time
import copy
import logging
from signal import signal, SIGINT

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

##################
from PIL import Image

##################

from worker import Worker, get_dummy_env
from models import MetaQNetController, ObsQNetController, ObservationOnlyQNet, ActionOnlyQNet, epsilon_greedy, epsilon_greedy_batch
from models import PPO
from memory import ReplayMemory, Buffer

class DoubleTrainer:
    def __init__(self, config):
        # signal(SIGINT, self.exit_handler)
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
        self.replay_memory_size = self.training_config['replay_memory_size']
        self.epsilon = self.training_config['epsilon']
        self.epoch = self.training_config['epoch']
        self.updates = self.training_config['updates']
        self.lr = self.training_config['learning_rate']
        self.gamma = self.training_config['gamma']
        self.lamda = 0.5
        self.batch_size = self.training_config['batch_size']
        
        # checkpoint setting
        self.model_checkpoint = self.model_config['model_checkpoint']
        self.model_checkpoint_path = self.model_config['model_checkpoint_path']


        self.n_meta_action = 2

        # get default shape of environment
        self.dummy_env = get_dummy_env(self.env_config['tower_config'], self.worker_id)
        self.dummy_env.reset()
        # print(self.dummy_env.observation_space)
        self.obs_shape = self.dummy_env.observation_space.shape
        # self.action_shape = self.dummy_env.action_space.n
        self.dummy_env.close()
        logger_message_obs = "observation.shape = "+str(self.obs_shape)
        # logger_message_action = "action.shape = "+str(self.action_shape)
        self.logger_epi.info(logger_message_obs)
        # self.logger_epi.info(logger_message_action)
        self.logger_default.info(logger_message_obs)
        # self.logger_default(logger_message_action)
        
        #  model setting
        self.ppomodel = PPO(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2]).cuda()
        self.buffer = Buffer(self.device)
        
        self.meta_Qlearner = MetaQNetController(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        self.target_meta_Qlearner = MetaQNetController(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        self.obs_Qlearner = ObservationOnlyQNet(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        self.target_obs_Qlearner = ObservationOnlyQNet(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        self.action_Qlearner = ActionOnlyQNet(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        self.target_action_Qlearner = ActionOnlyQNet(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        
        
        self.optimizer = optim.Adam(self.ppomodel.parameters(), lr=0.001)
        self.meta_optimizer = optim.Adam(params=self.meta_Qlearner.parameters(), lr=self.lr)
        self.obs_optimizer = optim.Adam(params=self.obs_Qlearner.parameters(), lr=self.lr)
        self.action_optimizer = optim.Adam(params=self.action_Qlearner.parameters(), lr=self.lr)
        # self.obsaction_optimzier = optim.Adam(params=self.)

        self.meta_memory = ReplayMemory(memory_size=self.training_config['replay_memory_size'])
        # self.obs_memory = ReplayMemory(memory_size=self.training_config['replay_memory_size'])
        # self.action_memory = ReplayMemory(memory_size=self.training_config['replay_memory_size'])
        self.obsaction_memory = ReplayMemory(memory_size=self.training_config['replay_memory_size'])

        self.writer = SummaryWriter('summaries/'+timestamp)
        self.write_hyperparameters(self.training_config)

    
        self.multiple_actions = np.zeros(self.n_workers)
        self.multiple_timestamp = np.zeros((2, self.n_workers))

        # worker initializaiton, start
        self.workers = []
        for i in range(1, self.n_workers+1):
            self.workers.append(Worker(self.env_config['tower_config'], self.worker_id+i))
        
        # reset environment
        for worker in self.workers:
            worker.parent_conn.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs = worker.parent_conn.recv()
    
        # for worker in self.workers:
        #     worker.parent_conn.send(('close', None))
    
    
    def reset_env(self, worker_id):
        worker = self.workers[worker_id]
        worker.parent_conn.send(("reset", None))
        state = worker.parent_conn.recv()
        state = state.transpose(2, 0, 1)
        self.multiple_history[worker_id][0:3] = state[:]
        for i in range(0, self.channel_size, 3):
            worker.parent_conn.send(("step", 0))
            next_state, _, _, _ = worker.parent_conn.recv()
            next_state = next_state.transpose(2, 0, 1)
            history_idx = int((i+3)%self.channel_size)
            self.multiple_history[worker_id][history_idx:history_idx+3] = next_state[:]
            self.multiple_next_history[worker_id][i:i+3] = next_state[:]   
    
    
    def optimizing(self):
        batch_idx = np.random.choice(self.meta_memory.size(), self.batch_size)
        meta_state, meta_action, meta_reward, meta_next_state, meta_done = self.meta_memory.uniform_sample(batch_idx)
        state, action, reward, next_state, done = self.obsaction_memory.uniform_sample(batch_idx)
        
        meta_q = self.meta_Qlearner(meta_state)
        meta_q_value = meta_q.gather(1, meta_action)
        target_meta_q = self.target_meta_Qlearner(meta_next_state).max(1)[0].unsqueeze(1)
        target_meta_value = meta_reward + self.gamma*target_meta_q*meta_done
        meta_loss = F.smooth_l1_loss(meta_q_value, target_meta_value)


        obs_idx = (meta_action==0).nonzero(as_tuple=True)[0] # tensor([0, 1, 2.....], device=)
        action_idx = (meta_action==1).nonzero(as_tuple=True)[0]        
        
        if len(obs_idx)!=0:
            obs_state = torch.index_select(state, 0, obs_idx)
            obs_action = torch.index_select(action, 0, obs_idx)
            obs_next_state = torch.index_select(next_state, 0, obs_idx)
            obs_reward = torch.index_select(reward, 0, obs_idx)
            obs_done = torch.index_select(done, 0, obs_idx)
            
            obs_q = self.obs_Qlearner(obs_state)
            obs_q_value = obs_q.gather(1, obs_action)
            obs_target_q = self.target_obs_Qlearner(obs_next_state).max(1)[0].unsqueeze(1)
            obs_target_value = obs_reward + self.gamma*obs_target_q*obs_done
            obs_loss = F.smooth_l1_loss(obs_q_value, obs_target_value)
        
        if len(action_idx)!=0:
            action_state = torch.index_select(state, 0, action_idx)
            action_action = torch.index_select(action, 0, action_idx)
            action_next_state = torch.index_select(next_state, 0, action_idx)
            action_reward = torch.index_select(reward, 0, action_idx)
            action_done = torch.index_select(done, 0, action_idx)
            
            action_q = self.action_Qlearner(action_state)
            action_q_value = action_q.gather(1, action_action)
            action_target_q = self.target_action_Qlearner(action_next_state).max(1)[0].unsqueeze(1)
            action_target_value = action_reward + self.gamma*action_target_q*action_done
            action_loss = F.smooth_l1_loss(action_q_value, action_target_value)
            
        if len(obs_idx)==0:
            loss = action_loss
        elif len(action_idx)==0:
            loss = obs_loss
        else:
            loss = F.smooth_l1_loss(obs_loss, action_loss)
        
        self.meta_optimizer.zero_grad()
        self.obs_optimizer.zero_grad()
        self.action_optimizer.zero_grad()
        
        meta_loss.backward()
        loss.backward()
        
        self.meta_optimizer.step()
        self.obs_optimizer.step()
        self.action_optimizer.step()
        
        
        # meta_state, meta_action, meta_reward, meta_next_state, meta_done = self.meta_memory.sample(self.batch_size)
        # obs_state, obs_action, obs_reward, obs_next_state, obs_done = obs_action_samples = self.obs_memory.sample(self.batch_size)
        # action_state, action_action, action_reward, action_next_state, action_done = action_action_samples = self.action_memory.sample(self.batch_size)
        
        # meta_q = self.meta_Qlearner(meta_state)
        # meta_q_value = meta_q.gather(1, meta_action)
        # target_meta_q = self.target_meta_Qlearner(meta_next_state).max(1)[0].unsqueeze(1)
        # target_meta_value = meta_reward + self.gamma*target_meta_q*meta_done
        # meta_loss = F.smooth_l1_loss(meta_q_value, target_meta_value)

        # obs_q = self.obs_Qlearner(obs_state)
        # obs_q_value = obs_q.gather(1, obs_action)

        # target_obs_q = self.target_obs_Qlearner(obs_next_state).max(1)[0].unsqueeze(1)
        # target_obs_value = obs_reward + self.gamma*target_obs_q*obs_done
        # obs_loss = F.smooth_l1_loss(obs_q_value, target_obs_value)
      
        # action_q = self.action_Qlearner(action_state)
        # action_q_value = action_q.gather(1, action_action)
        # target_action_q = self.target_action_Qlearner(action_next_state).max(1)[0].unsqueeze(1)
        # target_action_value = action_reward + self.gamma*target_action_q*action_done
        # action_loss = F.smooth_l1_loss(action_q_value, target_action_value)

    
        # self.meta_optimizer.zero_grad()
        # self.obs_optimizer.zero_grad()
        # self.action_optimizer.zero_grad()
        
        # meta_loss.backward()
        # obs_loss.backward()
        # action_loss.backward()
        
        # self.meta_optimizer.step()
        # self.obs_optimizer.step()
        # # self.action_optimizer.step()
        
    
    
    def one_step(self):
        tensor_multiple_history = torch.tensor(self.multiple_history, dtype=torch.float32, device=self.device)
        meta_q = self.meta_Qlearner(tensor_multiple_history)
        meta_actions_idx = epsilon_greedy_batch(meta_q, self.epsilon, n_actions=2)
        multiple_meta_actions = meta_actions_idx.tolist()
        one_step_result = []

        tmp_idx=0
        # select action (meta_action --> action --> env_action)
        for action_idx, worker in zip(multiple_meta_actions, self.workers):
            if action_idx==0: # obs
                obs_q = self.obs_Qlearner(tensor_multiple_history[tmp_idx])
                obs_action_idx = epsilon_greedy_batch(obs_q, self.epsilon, n_actions=2)
                if obs_action_idx==0:
                    env_action = 9 # left observation with jump
                else:
                    env_action = 15 # right observation with jump
                    
                self.multiple_actions[tmp_idx] = obs_action_idx
                    
            else: # action
                action_q = self.action_Qlearner(tensor_multiple_history[tmp_idx])
                action_action_idx = epsilon_greedy_batch(action_q, self.epsilon, n_actions=36)
                
                if action_action_idx >=14:
                    env_action = action_action_idx+2
                elif action_action_idx >=9:
                    env_action = action_action_idx+1
                else:
                    env_action = action_action_idx
                # env_action = env_action.cpu().data.numpy()
                env_action = env_action.item()
                self.multiple_actions[tmp_idx] = action_action_idx

            # self.multiple_actions[tmp_idx] = env_action
            worker.parent_conn.send(("step", env_action))
            tmp_idx+=1
        
        total_rewards = 0
        total_time_remaining = 0
        total_current_floor = 0
        worker_length = len(self.workers)
        # receive info from environment (action --> info)
        for idx, worker in enumerate(self.workers):
            next_state, reward, done, info = worker.parent_conn.recv()
            # info['total_keys']
            # info['time_remaining']
            # info['current_floor']
            next_state = next_state.transpose(2, 0, 1)

            self.multiple_history[idx] = np.roll(self.multiple_history[idx], -3, axis=0)
            self.multiple_next_history[idx] = np.roll(self.multiple_next_history[idx], -3, axis=0)
            self.multiple_history[idx][self.channel_size-3 : self.channel_size] = copy.deepcopy(self.multiple_next_history[idx][self.channel_size-6:self.channel_size-3])
            self.multiple_next_history[idx][self.channel_size-3 : self.channel_size] = next_state[:]

            self.meta_memory.put((self.multiple_history[idx], multiple_meta_actions[idx], reward, self.multiple_next_history[idx], done))
            self.obsaction_memory.put((self.multiple_history[idx], self.multiple_actions[idx], reward, self.multiple_next_history[idx], done))
            # if multiple_meta_actions[idx]==0: # obs
            #     self.obs_memory.put((self.multiple_history[idx], self.multiple_actions[idx], reward, self.multiple_next_history[idx], done))
            # else: # actions
            #     self.action_memory.put((self.multiple_history[idx], self.multiple_actions[idx], reward, self.multiple_next_history[idx], done))
        
            total_rewards+=reward
            total_time_remaining+=info['time_remaining']
            total_current_floor+=info['current_floor']
            self.multiple_timestamp[0][idx]+=1
            
            if done: #0 1 3 4 5 / 2 / 0, 1, 5 / 2, 3, 4
                self.multiple_timestamp[1][idx] = copy.deepcopy(self.multiple_timestamp[0][idx])
                self.multiple_timestamp[0][idx]=0
                self.reset_env(idx)
        
        return [total_rewards/worker_length, total_current_floor/worker_length, np.mean(self.multiple_timestamp[1])] 


    def image_check(self, image):
        # print(image[0:3].shape) 3, 12, 84, 84
        print(image[0][0:3])
        channel_image = image[0][0:3].transpose(1, 2, 0)
        print(channel_image)
        channel_image.astype(np.uint8)
        frame = Image.fromarray(channel_image)
        frame.convert('L').resize(size=(11, 11)).convert('P', palette=Image.ADAPTIVE, colors=8)
        
        
        print(image[0][0:3].shape)


    def sample(self):        
        action_mapping = {i:i*3 for i in range(12)}

        episode_infos = []
        rewards = np.zeros((4), dtype=np.float32)
            
        # self.image_check(self.multiple_history)
        
        for t in range(512):
            self.buffer.vis_obs[:, t] = self.multiple_history
            
            tensor_multiple_history = torch.tensor(self.multiple_history, dtype=torch.float32, device=self.device)
            prob = self.ppomodel.actor(tensor_multiple_history)
            value = self.ppomodel.critic(tensor_multiple_history).reshape(-1)
            
            self.buffer.values[:, t] = value.cpu().data.numpy()
            prob_categorical = Categorical(prob)
            actions = prob_categorical.sample()
            log_probs = prob_categorical.log_prob(actions)
            # print(actions)
            # print(Categorical(prob).log_prob(actions))
            self.buffer.actions[:, t] = actions.cpu().data.numpy()
            self.buffer.log_probs[:, t] = log_probs.cpu().data.numpy()
        
            # action execute
            for w, worker in enumerate(self.workers):
                worker.parent_conn.send(("step", action_mapping[self.buffer.actions[w, t]]))
                
            # receive
            for w, worker in enumerate(self.workers):
                next_state, reward, done, info = worker.parent_conn.recv()
                rewards[w]+=reward
                next_state = next_state.transpose(2, 0, 1)

                self.multiple_history[w] = np.roll(self.multiple_history[w], -3, axis=0)
                self.multiple_next_history[w] = np.roll(self.multiple_next_history[w], -3, axis=0)
                self.multiple_history[w][self.channel_size-3 : self.channel_size] = copy.deepcopy(self.multiple_next_history[w][self.channel_size-6:self.channel_size-3])
                self.multiple_next_history[w][self.channel_size-3 : self.channel_size] = next_state[:]
                
                self.buffer.rewards[w, t] = reward
                self.buffer.dones[w, t] = done
                                
                if done:
                    episode_infos.append(rewards[w])
                    self.reset_env(w)
                    rewards[w]=0.0
                    
        tmporal_tensor = torch.tensor(self.multiple_next_history, dtype=torch.float32, device=self.device)
        last_value = self.ppomodel.critic(tmporal_tensor).reshape(-1)

        self.buffer.calc_advantage(last_value.cpu().data.numpy(), self.gamma, self.lamda)
        
        return episode_infos

    @staticmethod
    def _normalize(adv: np.ndarray):

        return (adv - adv.mean()) / (adv.std() + 1e-8)


    def train_mini_batch(self, learning_rate, clip_range, beta, samples):
        sampled_return = samples['values'] + samples['advantages']
        # sampled_normalized_advantage = DoubleTrainer._normalize(samples['advantages']).unsqueeze(1).repeat(1, len(self.action_space_shape))
        # policy, value, _ = self.model(samples['vis_obs'] if self.vis_obs is not None else None,
        #                             samples['vec_obs'] if self.vec_obs is not None else None,
        #                             samples['hidden_states'] if self.use_recurrent else None,
        #                             self.device)
        tensor_vis_obs = torch.tensor(samples['vis_obs'], dtype=torch.float32, device=self.device)
        policy = self.ppomodel.actor(tensor_vis_obs)
        value = self.ppomodel.critic(tensor_vis_obs).reshape(-1)
        
        # Policy Loss
        # Retreive and process log_probs from each policy branch
        # log_probs = []
        # for i, policy_branch in enumerate(policy):
        #     log_probs.append(policy_branch.log_prob(samples['actions'][:, i]))
        # log_probs = torch.stack(log_probs, dim=1)
        log_probs_med = Categorical(policy)
        log_probs = log_probs_med.log_prob(samples['actions'])

        # Compute surrogates
        ratio = torch.exp(log_probs - samples['log_probs'])
        # surr1 = ratio * sampled_normalized_advantage
        surr1 = ratio * samples['advantages']
        # surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * sampled_normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * samples['advantages']

        policy_loss = torch.min(surr1, surr2).mean()

        # Value
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range,
                                                                      max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()
        
        entropy_bouns = log_probs_med.entropy().mean()
        # print(entropy_bouns)

        # Entropy Bonus
        # entropies = []
        # for policy_branch in policy:
        #     entropies.append(policy_branch.entropy())
        # entropy_bonus = torch.stack(entropies, dim=1).sum(1).reshape(-1).mean()
        # print(log_probs_med) 512, 12


        # Complete loss
        loss = -(policy_loss - 0.5 * vf_loss + beta * entropy_bouns)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ppomodel.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Monitor training statistics
        approx_kl_divergence = .5 * ((log_probs - samples['log_probs']) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_loss,
                vf_loss,
                loss,
                entropy_bouns,
                approx_kl_divergence,
                clip_fraction]



    def train_epochs(self, learning_rate, clip_range, beta):
        train_info = []
        
        for _ in range(4):
            mini_batch_generator = self.buffer.mini_batch_generator()
        for mini_batch in mini_batch_generator:
            res = self.train_mini_batch(learning_rate=learning_rate,
                                     clip_range=clip_range,
                                     beta = beta,
                                     samples=mini_batch)
            train_info.append(res)
        # Return the mean of the training statistics
        return np.mean(train_info, axis=0)            


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
            step_result = self.sample()


            self.buffer.prepare_batch_dict()
            
            training_stats = self.train_epochs(0.0001, 0.2, 0.01)
            print("learning")
            # if self.meta_memory.size() >= 40000:
            print("reward mean", np.mean(step_result))
            print("policy loss ", training_stats[0].item(), "  value loss ", training_stats[1].item(),
                  "  entropy ", training_stats[3].item(), "  approx_kl ", training_stats[4].item())
                
            #     self.optimizing()
            
            # if idx % self.model_checkpoint==0:
            #     torch.save(self.model, self.model_checkpoint_path+"-"+str(idx)+".pt")
            # if idx%100==0:
            #     self.logger_epi.info(f'rewards : {step_result[0]:.2f}  floor : {step_result[1]:.2f}  timestamp : {step_result[2]:.2f}')
            #     # print(f'rewards : {step_result[0]:.2f}  floor : {step_result[1]:.2f}  timestamp : {step_result[2]:.2f}')
            
            
    # def write_hyperparameters(self, config):
    #     """Writes hyperparameters to tensorboard"""
    #     for key, value in config.items():
    #         self.writer.add_text("Hyperparameters", key + " " + str(value))
            
            
    # def exit_handler(self):
    #     # print("Terminate")
    #     # self.close()
    #     # exit(0)