import torch
import torch.nn as nn
import torch.nn.functional as F

class PPO(nn.Module):
    def __init__(self, obs_h, obs_w, channels, n_actions):
        super(PPO, self).__init__()
        self.data = []
        self.observation = (obs_h, obs_w, channels)
        self.n_actions = n_actions
        
        self.input = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        
        # pytorch dqn
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        # self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(obs_h, 8, 2), 4, 2), 3, 1)
        # retro=True
        self.fc_actor = nn.Linear(in_features= 7*7*64, out_features=512)
        self.output_actor = nn.Linear(in_features=512, out_features=self.n_actions)
        self.fc_critic = nn.Linear(in_features = 7*7*64, out_features=512)
        self.output_critic = nn.Linear(in_features=512, out_features=1)

        
    def actor(self, x, softmax_dim = 0):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])
        
        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc_actor(x))
        # retro=False
        x = self.output_actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        
        return prob.squeeze()

    
    def critic(self, x):
        x = x.view(-1, self.observation[2], self.observation[0], self.observation[1])

        x = F.relu(self.input(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        
        x = F.relu(self.fc_critic(x))
        value = self.output_critic(x)

        return value
    
    
    
# def test():
    
#     ac = ActorCritic()