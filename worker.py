# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:24:26 2021

@author: 6794c
"""
import torch.multiprocessing as mp
from obstacle_tower_env import ObstacleTowerEnv


def get_dummy_env(env_config, worker_id):
    env = ObstacleTowerEnv(environment_filename="ObstacleTower\\ObstacleTower.exe",
                           worker_id=worker_id,
                           retro=True,
                           timeout_wait=60,
                           realtime_mode=False,
                           config=env_config,
                           greyscale=False)
    
    return env

def runner(conn, env_config, worker_id):
    env = ObstacleTowerEnv(environment_filename="ObstacleTower\\ObstacleTower.exe",
                           worker_id=worker_id,
                           retro=True,
                           timeout_wait=60,
                           realtime_mode=True,
                           config=env_config,
                           greyscale=False)
    
    history = []
    next_history = []
    
    state = env.reset()
    history.append(state)
    
    while True:
        try:
            cmd, data = conn.recv()
            if cmd=="step":
                next_state, reward, done, info = env.step(data)
                conn.send(env.step(data))
            elif cmd=="reset":
                conn.send(env.reset(data))
            elif cmd=="close":
                conn.send(env.close())
                conn.close()
                break
            else:
                raise NotImplementedError
                
        except:
            break


class Worker:
    def __init__(self, env_config, worker_id):

        self.parent_conn, child_conn = mp.Pipe()
        self.process = mp.Process(target=runner,
                                  args=(child_conn, env_config, worker_id))
        self.process.start()