# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:39:50 2021

@author: 6794c
"""
from utilset import get_logger_epi, get_logger_default, load_configuration
from trainer import DoubleTrainer

def main():

    config = load_configuration('config_ppo.yaml')
    log_path = config['default']['log_path']
    
    logger_epi = get_logger_epi(log_path)
    logger_default = get_logger_default(log_path)
    logger_epi.info('training start')
    logger_default.info('training start')
    
    trainer = DoubleTrainer(config)
    trainer.training()
    
if __name__ == '__main__':
    main()