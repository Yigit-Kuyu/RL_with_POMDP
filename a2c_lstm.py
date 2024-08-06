import gymnasium as gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Bernoulli
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dimensions 
input_dim = 2
fc1_output_dim = 64
hidden_size_policy = 128
hidden_size_value = 256
fc2_output_dim_policy = 1
fc2_output_dim_value = 1

batch_size = 1
sequence_length = 1


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_output_dim)
        self.lstm = nn.LSTM(fc1_output_dim, hidden_size_policy, batch_first=True)
        self.fc2 = nn.Linear(hidden_size_policy, fc2_output_dim_policy)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        x = self.relu(x)
        x = self.sigmoid(self.fc2(x))
        return x, hidden

    def select_action(self, state, hidden):
        with torch.no_grad():
            prob, hidden = self.forward(state, hidden)
            b = Bernoulli(prob)
            action = b.sample()
        return action.item(), hidden


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_output_dim)
        self.lstm = nn.LSTM(fc1_output_dim, hidden_size_value, batch_first=True)
        self.fc2 = nn.Linear(hidden_size_value, fc2_output_dim_value)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        x = self.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        x = self.relu(x)
        x = self.fc2(x)
        return x, hidden

def preprocess_state(state):
    return np.array([state[0], state[2]])  # Extract position and angle only

def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def update_policy(policy, states_tensor, actions_tensor, rewards_tensor, advantage, optim):
    p_hx = torch.zeros((batch_size, sequence_length, hidden_size_policy)).to(device) # Hidden state for the PolicyNetwork (for short-term memory)
    p_cx = torch.zeros((batch_size, sequence_length, hidden_size_policy)).to(device) # Cell state for the PolicyNetwork(for long-term memory)
    prob, _ = policy(states_tensor, (p_hx, p_cx))
    prob = prob.squeeze(0)
    b = Bernoulli(prob)
    log_prob = b.log_prob(actions_tensor)
    loss = -log_prob * advantage
    loss = loss.mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

def update_value_network(value, states_tensor, rewards_tensor, value_optim):
    v_hx = torch.zeros((batch_size, sequence_length, hidden_size_value)).to(device) # hidden for the ValueNetwork (for short-term memory)
    v_cx = torch.zeros((batch_size, sequence_length, hidden_size_value)).to(device)# cell states for the ValueNetwork (for long-term memory)
    v, _ = value(states_tensor, (v_hx, v_cx))
    v = v.squeeze(0)
    value_loss = F.mse_loss(rewards_tensor, v)
    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()

def train(env, policy, value, optim, value_optim, gamma=0.99, num_epochs=1000):
    for epoch in range(num_epochs):
        state = env.reset() # Original state: Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
        state=state[0]
        state = preprocess_state(state)
        episode_reward = 0

        
        #Hidden and cell states for the LSTM in PolicyNetwork
        p_hx = torch.zeros((batch_size, sequence_length, hidden_size_policy)).to(device)
        p_cx = torch.zeros((batch_size, sequence_length, hidden_size_policy)).to(device)
        

        rewards = []
        actions = []
        states = []

        for time_steps in range(200):
            states.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            action, (p_hx, p_cx) = policy.select_action(state_tensor, (p_hx, p_cx))
            actions.append(int(action))
            next_state, reward, done, _, _ = env.step(int(action))
            next_state = preprocess_state(next_state)
            episode_reward += reward
            state = next_state
            rewards.append(reward)
            env.render()

            # An episode ends when: 
            # 1) the pole is more than 15 degrees from vertical; or 
            # 2) the cart moves more than 2.4 units from the center.
            if done:
                break

        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
        actions_tensor = torch.FloatTensor(actions).unsqueeze(1).to(device)
        rewards_tensor = returns.unsqueeze(1).to(device)
        
        # Use, torch.no_grad() because, when you are not updating the model parameters 
        # but only evaluating the model's performance, such as computing the value estimates.
        with torch.no_grad():
            # batch size of 1, sequence length of 1, and hidden state size of 256
            v_hx = torch.zeros((batch_size, sequence_length, hidden_size_value)).to(device) # hidden for the ValueNetwork (for short-term memory)
            v_cx = torch.zeros((batch_size, sequence_length, hidden_size_value)).to(device) # cell states for the ValueNetwork (for long-term memory)
            v, _ = value(states_tensor, (v_hx, v_cx))
            v = v.squeeze(0)
            advantage = rewards_tensor - v

        update_policy(policy, states_tensor, actions_tensor, rewards_tensor, advantage, optim)
        update_value_network(value, states_tensor, rewards_tensor, value_optim)

        if epoch % 10 == 0:
            print(' Training Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
            torch.save(policy.state_dict(), 'lstm_network.pkl')


def test(env, policy, num_episodes=10):
    total_rewards = []
    policy.load_state_dict(torch.load('a2c_lstm_network.pkl'))  # Load the trained model weights
    num_comp_ep=0
    for ep in range(num_episodes):
        state = env.reset()
        state=state[0]
        state = preprocess_state(state)
        episode_reward = 0
        done = False

        
         #Hidden and cell states for the LSTM in PolicyNetwork
        p_hx = torch.zeros((batch_size, sequence_length, hidden_size_policy)).to(device)
        p_cx = torch.zeros((batch_size, sequence_length, hidden_size_policy)).to(device)
        

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            action, (p_hx, p_cx) = policy.select_action(state_tensor, (p_hx, p_cx))
            next_state, reward, done, _, _ = env.step(int(action))
            next_state = preprocess_state(next_state)
            episode_reward += reward
            state = next_state
            if done:
               num_comp_ep+=1
               print("number of completed episodes: ", num_comp_ep)
            print(' Testing Epoch:{}, episode reward is {}'.format(ep, episode_reward))

        total_rewards.append(episode_reward)
        print("Percentage of completed episodes: ", (num_comp_ep/num_episodes)*100)
    return total_rewards



if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    policy = PolicyNetwork().to(device)
    value = ValueNetwork().to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    value_optim = torch.optim.Adam(value.parameters(), lr=3e-4)

    # Training phase
    train(env, policy, value, optim, value_optim)

    # Testing phase
    test_rewards = test(env, policy)
    print('Average test reward:', np.mean(test_rewards))








