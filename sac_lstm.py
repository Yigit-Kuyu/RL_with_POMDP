import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from itertools import count
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        

    def forward(self, state, hidden):
        batch_size = state.size(0) # optional
        seq_len = state.size(1) if state.dim() > 2 else 1 # optional
        
        if state.dim() == 2:
            state = state.unsqueeze(1)  
        
        x1 = self.relu(self.fc1(state))
        x2, hidden = self.lstm(x1, hidden)
        x3 = self.relu(x2)
        logits = self.fc2(x3)
        return logits, hidden

    def sample(self, state, hidden):
        logits, hidden = self.forward(state, hidden)
        probs = self.softmax(logits)
        dist = Categorical(probs)
        action = dist.sample().unsqueeze(-1) 
        log_prob = dist.log_prob(action.squeeze(-1))
        return action, log_prob, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(device))

    

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


    def forward(self, state, action, hidden):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        x, hidden = self.lstm(x, hidden) # LSTM layer is expecting a 3D input (batch_size, sequence_length, input_size) in x
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.relu(x)
        x = self.fc2(x)
        return x, hidden

    def q_values(self, state, action, hidden):
        q1, hidden1 = self.forward(state, action, hidden)
        q2, hidden2 = self.forward(state, action, hidden)
        return q1, q2, hidden1, hidden2
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(1, batch_size, self.lstm.hidden_size).to(device))


class ReplayMemory:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.memory_capacity

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class SACAgent:
    def __init__(self, input_dim, action_dim, hidden_dim, memory_capacity, batch_size,
                 gamma, tau, num_updates, policy_freq, alpha):
        self.actor = Actor(input_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(input_dim, hidden_dim, action_dim).to(device)
        self.critic_target = Critic(input_dim, hidden_dim, action_dim).to(device)
        self.value = Critic(input_dim, hidden_dim, 0).to(device)
        self.value_target = Critic(input_dim, hidden_dim, 0).to(device)

        self.memory = ReplayMemory(memory_capacity, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.policy_freq = policy_freq

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-4)

        self.hard_update(self.critic_target, self.critic)
        self.hard_update(self.value_target, self.value)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = alpha

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, batch):
        for update_step in range(self.num_updates):
            state, action, reward, next_state, mask = batch
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.LongTensor(action).to(device).unsqueeze(-1)
            reward = torch.FloatTensor(reward).to(device)
            mask = torch.FloatTensor(mask).to(device)

            action_one_hot = torch.zeros(state.size(0), 2).to(device).scatter_(1, action, 1)

            with torch.no_grad():
                next_hidden = self.actor.init_hidden(state.size(0))
                next_action, next_log_prob, next_hidden = self.actor.sample(next_state, next_hidden)
                next_action_one_hot = torch.zeros(state.size(0), 2).to(device).scatter_(1, next_action.squeeze(-1).long(), 1)
                q1_target, q2_target, _, _ = self.critic_target.q_values(next_state, next_action_one_hot, next_hidden)
                q_target = torch.min(q1_target, q2_target)
                value_target = reward + mask * self.gamma * (q_target - next_log_prob)
            
            hidden = self.critic.init_hidden(state.size(0))
            q1, q2, _, _ = self.critic.q_values(state, action_one_hot, hidden)
            critic_loss = F.mse_loss(q1, value_target) + F.mse_loss(q2, value_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if update_step % self.policy_freq == 0:
                hidden = self.actor.init_hidden(state.size(0))
                action, log_prob, hidden = self.actor.sample(state, hidden)
                action_one_hot = torch.zeros(state.size(0), 2).to(device).scatter_(1, action.squeeze(-1).long(), 1)
                q1, q2, _, _ = self.critic.q_values(state, action_one_hot, hidden)
                q_value = torch.min(q1, q2)
                actor_loss = (self.alpha * log_prob - q_value).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

                self.soft_update(self.critic_target, self.critic)
                self.soft_update(self.value_target, self.value)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        hidden = self.actor.init_hidden(1)
        action, _, hidden = self.actor.sample(state, hidden)
        return action.cpu().detach().numpy()[0]

    def step(self):
        batch = self.memory.sample()
        self.learn(batch)

    def save(self):
        torch.save(self.actor.state_dict(), "sac_actor.pth")
        torch.save(self.critic.state_dict(), "sac_critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("sac_actor.pth"))
        self.critic.load_state_dict(torch.load("sac_critic.pth"))



def preprocess_state(state):
    return np.array([state[0], state[2]])  # Extract position and angle only


def train(env, agent, num_episodes=200):
    reward_list = []
    avg_reward_list = []
    avg_reward_deque = deque(maxlen=100)

    for i in range(num_episodes):
        state = env.reset()
        state=state[0]
        state = preprocess_state(state)
        episode_reward = 0
        done = False
        # Hidden and cell states for the LSTM in policy network
        p_hx, p_cx = torch.zeros((1, 1, 128)).to(device), torch.zeros((1, 1, 128)).to(device)

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            action, _, (p_hx, p_cx) = agent.actor.sample(state_tensor, (p_hx, p_cx))
            next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0][0][0])
            next_state = preprocess_state(next_state)
            mask = float(not done)

            agent.memory.push((state, action, reward, next_state, mask))
            state = next_state
            episode_reward += reward

            if len(agent.memory) > agent.memory.batch_size:
                agent.step()

        reward_list.append(episode_reward)
        avg_reward_deque.append(episode_reward)
        avg_reward_list.append(np.mean(avg_reward_deque))

        if i % 10 == 0:
            print(f"Training Episode {i}, Reward: {episode_reward}, Average Reward: {np.mean(avg_reward_deque)}")

    return reward_list, avg_reward_list


def test(env, agent, num_episodes=20):
    agent.load()  # Load the saved network
    total_rewards = []
    num_comp_ep = 0

    for ep in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = preprocess_state(state)
        episode_reward = 0
        done = False
        
        # Initialize hidden states for actor
        hidden = agent.actor.init_hidden(1)

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            action, _, hidden = agent.actor.sample(state_tensor, hidden)
            next_state, reward, done, _, _ = env.step(action.cpu().numpy()[0][0][0])
            next_state = preprocess_state(next_state)
            
            state = next_state
            episode_reward += reward

            if done:
                num_comp_ep += 1
                print("number of completed episodes: ", num_comp_ep)

        print(' Testing Epoch:{}, episode reward is {}'.format(ep, episode_reward))
        total_rewards.append(episode_reward)
        
    print("Percentage of completed episodes: ", (num_comp_ep/num_episodes)*100)
    
    return total_rewards

# Add this method to your SACAgent class
def load(self):
    self.actor.load_state_dict(torch.load("sac_actor.pth"))
    self.critic.load_state_dict(torch.load("sac_critic.pth"))
    print("Model loaded successfully.")

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    input_dim = 2
    action_dim = env.action_space.n
    hidden_dim = 128
    memory_capacity = 10000
    batch_size = 64
    gamma = 0.99
    tau = 0.005
    num_updates = 1
    policy_freq = 2
    alpha = 0.2

    agent = SACAgent(input_dim, action_dim, hidden_dim, memory_capacity, batch_size,
                     gamma, tau, num_updates, policy_freq, alpha)
    
    # Training phase
    train_rewards, avg_train_rewards = train(env, agent, num_episodes=200)

    # Save the trained model
    agent.save()

    # Plot training results
    plt.plot(train_rewards, label='Train Rewards')
    plt.plot(avg_train_rewards, label='Average Train Rewards')
    plt.legend()
    plt.show()
    
    
    # Testing phase
    test_rewards = test(env, agent, num_episodes=20)

    print(f"Average test reward: {sum(test_rewards) / len(test_rewards)}")
    print(f"Max test reward: {max(test_rewards)}")
    print(f"Min test reward: {min(test_rewards)}")

    # Plot testing results
    plt.plot(test_rewards, label='Test Rewards')
    plt.legend()
    plt.show()



