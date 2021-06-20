# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:05:25 2021

@author: 6794c
"""

import logging
from logging.handlers import RotatingFileHandler

import yaml

def get_logger_epi(log_dictname):
    logger = logging.getLogger('obstacle_logger_epi')
    
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    log_filename = log_dictname+'/obstacle_epi.log'
    # file_handler = logging.handlers.RotatingFileHandler(filename='log_fixed/obstacle_epi.log',
    #                                                     maxBytes=50*1024*1024,
    #                                                     backupCount=1000)
    file_handler = logging.handlers.RotatingFileHandler(filename=log_filename,
                                                        maxBytes=50*1024*1024,
                                                        backupCount=1000)
    
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger

def get_logger_default(log_dictname):
    logger = logging.getLogger('obstacle_logger_default')
        
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    
    # logging.basicConfig(filemode='w')
    
    # stream_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler('obstacle.log')
    log_filename = log_dictname+'/obstacle.log'
    # file_handler = logging.handlers.RotatingFileHandler(filename='log_fixed/obstacle.log',
    #                                                     maxBytes=50*1024*1024,
    #                                                     backupCount=1000)
    file_handler = logging.handlers.RotatingFileHandler(filename=log_filename,
                                                        maxBytes=50*1024*1024,
                                                        backupCount=1000)
    
    # stream_handler.setFormatter(log_formatter)
    file_handler.setFormatter(log_formatter)
    # __logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
#     __logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    
    return logger


def load_configuration(config_file):
    with open(config_file) as f:
        # full_config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.safe_load(f)
        
        f.close()
        
    return config