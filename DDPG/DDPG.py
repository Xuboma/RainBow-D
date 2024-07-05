import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from runner import Runner
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import random

class RunnerModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(RunnerModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出范围在[-1, 1]之间


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_size, action_size, replay_buffer_capacity=10, continuous_action_space=(-1.0, 1.0)):
        self.actor = RunnerModel(state_size, action_size)
        self.actor_target = RunnerModel(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.continuous_action_space = continuous_action_space

        self.critic = nn.Sequential(
            nn.Linear(state_size + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_target = nn.Sequential(
            nn.Linear(state_size + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def select_action(self, state, velocity_space, exploration_noise=0.3):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action_values = self.actor(state).numpy()

        noise = exploration_noise * np.random.randn(len(velocity_space))
        action_values += noise

        # 映射到连续动作空间
        mapped_actions = self.map_discrete_to_continuous(action_values, velocity_space)

        return mapped_actions

    def map_discrete_to_continuous(self, discrete_actions, discrete_action_space):
        continuous_action_space = np.linspace(*self.continuous_action_space, len(discrete_action_space))
        mapped_actions = discrete_actions * (continuous_action_space.max() - continuous_action_space.min()) / 2.0
        return mapped_actions

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, state, action, reward, next_state, done, batch_size=5, gamma=0.99):
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < batch_size:
            return

        for _ in range(5):  # Perform 5 updates using a batch of 5 samples
            batch = self.replay_buffer.sample(batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.tensor(state_batch, dtype=torch.float32)
            action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(dim=-1)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
            next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
            done_batch = torch.tensor(done_batch, dtype=torch.float32)

            actor_target_values = self.actor_target(next_state_batch)
            max_action_value_index = torch.argmax(actor_target_values, dim=1, keepdim=True)
            next_state_values = actor_target_values.gather(1, max_action_value_index)

            critic_target = reward_batch + gamma * (1 - done_batch) * self.critic_target(
                torch.cat([next_state_batch, next_state_values], dim=-1)
            )

            critic_loss = nn.MSELoss()(self.critic(torch.cat([state_batch, action_batch], dim=-1)),
                                       critic_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(torch.cat([state_batch, action_batch], dim=-1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target, 0.001)
            self.soft_update(self.critic, self.critic_target, 0.001)


class RunningEnvironment:
    def __init__(self):
        self.distance = 800
        self.v = 3.5
        self.runner = Runner(55, self.distance / 1000)
        WS = self.runner.W_s(self.v)
        total_w1 = self.runner.total_w(self.distance / 1000)
        common_person_ratio = ((self.distance / self.v) * (WS + 110)) / total_w1
        print('Normal person consumes {:.5%} stamina running {}m'.format(common_person_ratio, self.distance))
        self.STAMINA = total_w1
        print('Total stamina:', self.STAMINA)
        self.state_size = 2
        self.current_distance = 0
        self.remaining_endurance = 1
        self.continuous_action_space = [2.0, 7.0]  # 根据你的需求设置合适的范围

    def reset(self):
        self.current_distance = 0
        self.remaining_endurance = 1
        return np.array([0, self.remaining_endurance])

    def get_velocity_space(self):
        if np.any(self.current_distance < 15):
            velocity_space = [2, 3, 4, 5]
        elif np.any((15 <= self.current_distance) & (self.current_distance < 600)):
            velocity_space = [3, 4, 5, 6]
        else:
            velocity_space = [4, 5, 6, 7]
        return velocity_space

    def calculate_reward(self, sta):
        return sta * sta

    def step(self, action):
        velocity = action
        distance_covered = action
        self.current_distance += distance_covered

        # 反映射到离散动作空间
        discrete_action_space = self.get_velocity_space()
        mapped_action = (action - self.continuous_action_space[0]) / (self.continuous_action_space[1] - self.continuous_action_space[0]) * 2.0
        mapped_action = np.clip(mapped_action, -1.0, 1.0)
        mapped_action = np.interp(mapped_action, np.linspace(-1.0, 1.0, len(discrete_action_space)), discrete_action_space)

        WS = self.runner.W_s(mapped_action) + 110
        sta_radio = self.remaining_endurance - WS / self.STAMINA
        reward = self.calculate_reward(WS)
        done = self.current_distance >= self.distance
        next_state = np.array([sta_radio, self.current_distance])

        return next_state, reward, done, WS


def train_ddpg(agent, env, episodes=200, save_path=None):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        runtime = 0
        st = env.STAMINA
        v_arry = []
        dis = []

        for i in range(200):
            velocity_space = env.get_velocity_space()
            action = agent.select_action(state, velocity_space)

            next_state, reward, done, W_S = env.step(action)
            v_arry.insert(len(v_arry), action)
            dis.insert(i, next_state[1])

            st = st - W_S
            st_radio = st / env.STAMINA
            if np.all(next_state[1] <= 15) and np.all(1 - st_radio > 0.014):
                reward = 0
                # 修改为
            elif np.any(15 < next_state[1] < 600) and np.any(1 - st_radio > 0.105):

                reward = 0

            agent.train(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            runtime = runtime + 1
            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        print(runtime)

        if save_path is not None and (episode + 1) % 20 == 0:
            agent.save_model(save_path)
            print(f"Model saved at episode {episode + 1}")


def test_ddpg(agent, env):
    agent.load_model("trained_actor_model.pth")

    state = env.reset()
    total_reward = 0
    runtime = 0
    v_arry = []
    dis = []

    while True:
        velocity_space = env.get_velocity_space()
        action = agent.select_action(state, velocity_space)
        next_state, reward, done, _ = env.step(action)
        v_arry.append(action)
        dis.append(next_state[1])

        state = next_state
        total_reward += reward
        runtime = runtime + 1
        if done:
            break

    print(f"Test Complete, Total Reward: {total_reward}")
    print(f"Runtime: {runtime}")
    print(v_arry)


env = RunningEnvironment()
state_size = env.state_size
action_size = len(env.get_velocity_space())
continuous_action_space = (-1.0, 1.0)
agent = DDPGAgent(state_size, action_size, continuous_action_space=continuous_action_space)
train_ddpg(agent, env, save_path="trained_actor_model.pth")
test_ddpg(agent, env)
