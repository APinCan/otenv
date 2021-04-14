from training import training
from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation

import logging


# https://velog.io/@devmin/first-python-logging
# https://inma.tistory.com/136
def __get_logger():
    __logger = logging.getLogger('obstacle_logger')
        
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    
    # logging.basicConfig(filemode='w')
    
    # stream_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler('obstacle.log')
    file_handler = logging.handlers.RotatingFileHandler(filename='obstacle.log',
                                                        maxBytes=50*1024*1024,
                                                        backupCount=1000)
    
    # stream_handler.setFormatter(log_formatter)
    file_handler.setFormatter(log_formatter)
    
    # __logger.addHandler(stream_handler)
    __logger.addHandler(file_handler)
    
#     __logger.addHandler(stream_handler)
    __logger.setLevel(logging.INFO)
    
    return __logger
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
 9: [0, 1, 1, 0],
 10: [0, 1, 1, 1],
 11: [0, 1, 1, 2],
 12: [0, 2, 0, 0],
 13: [0, 2, 0, 1],
 14: [0, 2, 0, 2],
 15: [0, 2, 1, 0],
 16: [0, 2, 1, 1],
 17: [0, 2, 1, 2],
 18: [1, 0, 0, 0],
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

def main():
    logger = __get_logger()
    
    logger.info('Training start')
    
    params = {
        'gamma':0.98,
        'lambda':0.95,
        'lr':0.0005,
        'epsilon':0.1,
        'episodes':3000,
        'seed':42,
        'k_epoch':3,
        'coef_value':0.5,
        'coef_entropy':0.01,
        'time_horizon':20
    }
    
    tower_config = {
        'train_mode':1,
        'tower-seed':-1,
        'starting-floor':0,
        'total-floors':100,
        'dense-reward':1,
        'lighting-type':1,
        'visual-theme':1,
        'agent-perspective':1,
        'allowed-rooms':2,
        'allowed-modluels':2,
        'allowed-floors':2,
        'default-theme':0
    }

    env = ObstacleTowerEnv(retro=True, realtime_mode=True)

    training(params, env)
    
    
if __name__=="__main__":
    main()
# main()

