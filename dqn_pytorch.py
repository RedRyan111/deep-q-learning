import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym
from collections import deque

class act_model(nn.Module):
    def __init__(self,inp,hidden,output):
        super(act_model, self).__init__()
        self.fc1 = nn.Linear(inp, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output, bias=True)
        self.fc12 = nn.LeakyReLU()
        
        self.memory = deque(maxlen=500)
        
        self.gamma = .95
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = .001
        self.epsilon_decay = .995
        
        self.mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),lr=.001)
        
    def action(self,state):
        if(random.random() <= self.epsilon):
            return np.random.choice(out, 1)[0]
        else:
            q_values = self.forward(state)
            return np.argmax(q_values.detach().numpy())  #Q(s,a)
            
    def memorize(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def replay(self,batch_size):
        if(len(self.memory) < batch_size): return 0 
        minibatch = random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            target = reward
            if not done:
                q_values = self.forward(next_state)
                target = (reward + self.gamma*np.amax(q_values.detach().numpy()))
            target_f = self.forward(state)
            target_f[action] = target
            target_g = self.forward(state)

            self.zero_grad()
            self.optimizer.zero_grad()
            
            loss = self.mse(target_g,target_f)
            loss.backward(retain_graph=True)
            self.optimizer.step() 
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self,PATH):
        torch.save(self, PATH)

    def save(self,PATH):
        model = torch.load(PATH)
        return model
        
    def forward(self,x):
        out = self.fc12(self.fc1(x))
        out = self.fc12(self.fc2(out))
        out = self.fc3(out)
        
        return out

inp = 4
hid = 24
out = 2
model = act_model(inp,hid,out)
env = gym.make('CartPole-v0')
epochs = 2000
batch_size = 50

for epoch in range(epochs):
    state = env.reset()
    for t in range(200):
        state = torch.FloatTensor(state)
        action = model.action(state)
        next_state, reward, done, info = env.step(action)
        reward = reward if not done else -10
        
        next_state = torch.FloatTensor(next_state)
        model.memorize(state,action,reward,next_state,done)
        state = next_state
        
        model.replay(batch_size)

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(epoch, epochs, t, model.epsilon))
            break   
env.close()

