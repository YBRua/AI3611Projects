import math

import torch.nn as nn


def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)
    
    
class Crnn(nn.Module):
    def __init__(self, num_freq, class_num):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     class_num: int, the number of output classes
        ##############################
        pass

    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        pass
        
    def forward(self, x): 
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }
