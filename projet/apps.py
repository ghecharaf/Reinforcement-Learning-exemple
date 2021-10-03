import gym
import math
import random
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

is_ipython='inline'in matplotlib.get_backend()
if is_ipython: from IPython import display


class DQN(nn.Module):

    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=24)
        self.out = nn.Linear(in_features=24, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = self.out(t)
        return t



Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def mem_size(self):
        return len(self.memory)


class EpsilonGreedyStrategy():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

class Agent():

    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        return policy_net(state).argmax(dim=1).to(self.device) # exploit

class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.cpt=0

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return reward


    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device) # add a batch dimensi

def plot(values, moving_avg_period, eps, mem_count):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    moving_avg = get_moving_average(moving_avg_period, values)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1], ", eps: ", eps, ", mem: ", mem_count)
    if is_ipython: display.clear_output(wait=True)

def plot2(values,eps):
    plt.figure(2)
    plt.clf()
    plt.title('Test')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    plt.pause(0.001)
    if is_ipython: display.clear_output(wait=True)
    print("Episode", eps, "\n",  ", Reward: ", values)




def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def plt_clear():
    for f in plt.get_fignums():
        plt.figure(f)
        plt.close()

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

# commandline hyperparameters

# parser = OptionParser("usage: %%prog [options]")

from django.shortcuts import render as rendd

from django.http import HttpResponse


def hd(request):

    context = {"var" : output}

    return rendd(request,"test.html",context)

from django.template.loader import render_to_string


def test():
    global output
    output = []
    print("yesssssssss")

    DEFAULT_BATCH_SIZE    = 2048
    DEFAULT_GAMMA         = 0.999
    DEFAULT_EPS_DECAY     = 0.001
    DEFAULT_TARGET_UPDATE = 10
    DEFAULT_MEMORY_SIZE   = 100000
    DEFAULT_LEARNING_RATE = 0.0001
    DEFAULT_NUM_EPISODES  = 3000



    #hyperparameters hado li lazam nbalhom f tranning bah el model yatsagam

    ##################################################################################
    batch_size=2048
    gamma=0.999
    eps_start=1.0
    eps_end=0.01
    eps_decay=0.0007
    target_update=10
    memory_size=500000
    lr=0.0005
    num_episodes=3

    print ("arguments: batch_size=%d gamma=%f eps_start=%f eps_end=%f eps_decay=%f target_update=%d memory_size=%d lr=%f num_episodes=%d" \
           % (batch_size, gamma, eps_start, eps_end, eps_decay, target_update, memory_size, lr, num_episodes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = CartPoleEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(strategy, em.num_actions_available(), device)




    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    print("trueeeeeeeeee")
    #hna dir el modele dyalak
    policy_net.load_state_dict(torch.load("projet/static/Mo.pth",map_location='cpu'))
    policy_net.eval()




    episode_durations = []

    for episode in range(num_episodes):

        em.reset()
        state = em.get_state()
        cpt=0
        em.done=False
        while not em.done:
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            cpt=cpt+reward
            #print(reward)
            next_state = em.get_state()
            state = next_state
            if em.done:
                episode_durations.append(cpt)
                s = "Episode "+str(episode)+"   Reward: "+str(cpt)
                output.append(s)
                print("Episode", episode, "   Reward: ", cpt)
                episode_durations.append(cpt)
                break
                #plot2(cpt,episode)

    em.close()
    print("la somme est",sum(episode_durations)/100)
    context = {"var" : output}
    return context


    #torch.save(policy_net.state_dict() ,"C:/Users/islam/Music/islam.pth")

