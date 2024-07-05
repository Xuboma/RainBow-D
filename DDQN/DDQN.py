import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from runner import Runner
import torch.nn.functional as F
import random
# import matplotlib.pyplot as plt
import pandas as pd
# import os

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

class RunnerModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(RunnerModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    def __init__(self, state_size, action_size, replay_buffer_capacity=10000, batch_size=64):
        self.q_network = RunnerModel(state_size, action_size)
        self.target_q_network = RunnerModel(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.batch_size = batch_size

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()  # Set the model to evaluation mode

    def select_action(self, state, velocity_space, exploration_noise=0.3):
        state = torch.tensor(state, dtype=torch.float32)

        action_values = self.q_network(state)

        action_probabilities = F.softmax(action_values, dim=-1)

        max_index = torch.argmax(action_probabilities[:len(velocity_space)])
        action = velocity_space[max_index]

        return action

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, state, action, reward, next_state, done, gamma=0.99):
        state[1] = state[1] / 800
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(dim=-1)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(dim=0)
        q_values = self.q_network(state+action)
        # q_values = self.q_network(torch.cat([state[:, 1], action], dim=-1))

        with torch.no_grad():
            next_q_values_target = self.target_q_network(next_state)

        target_actions = torch.argmax(self.q_network(next_state), dim=-1, keepdim=True)
        target_q_values = next_q_values_target
        target_q_values = reward + gamma * (1 - done) * target_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network, self.target_q_network, 0.001)

class RunningEnvironment:
    def __init__(self):
        self.distance = 800
        self.v = 3.5
        self.runner = Runner(60, self.distance / 1000)
        WS = self.runner.W_s(self.v)
        total_w1 = self.runner.total_w(self.distance / 1000)
        common_person_ratio = ((self.distance / self.v) * (WS + 110)) / total_w1
        print('Normal person consumes {:.5%} stamina running {}m'.format(common_person_ratio, self.distance))
        self.STAMINA = total_w1
        print('Total stamina:', self.STAMINA)
        self.state_size = 2
        self.current_distance = 0
        self.remaining_endurance = 100

    def reset(self):
        self.current_distance = 0
        self.remaining_endurance = 100
        return np.array([0, self.remaining_endurance])

    def get_velocity_space(self):
        if self.current_distance < 15:
            velocity_space = [2, 3, 4, 5]
        elif 15 <= self.current_distance < 600:
            velocity_space = [4, 5, 6, 7]
        else:
            velocity_space = [6, 7, 8, 9]
        return velocity_space

    def calculate_reward(self, sta):
        return sta * sta

    def step(self, action):
        velocity = action
        distance_covered = action
        self.current_distance += distance_covered
        WS = self.runner.W_s(velocity) + 110
        # 对状态进行修改加入剩余体力值
        sta_radio = self.remaining_endurance - WS / self.STAMINA
        reward = self.calculate_reward(WS)
        done = self.current_distance >= self.distance
        next_state = np.array([sta_radio, self.current_distance])
        return next_state, reward, done, WS

def train_ddqn(agent, env, episodes=100, save_path=None):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        runtime = 0
        st = env.STAMINA
        
        v_arry = []
        for i in range(200):
            velocity_space = env.get_velocity_space()
            action = agent.select_action(state, velocity_space)
            v_arry.insert(len(v_arry), action)
            next_state, reward, done, W_S = env.step(action)
            st = st - W_S
            st_radio = st / env.STAMINA

            # 20 50 30
            if next_state[1] <= 15 and 1 - st_radio > 0.0075:
                reward = 0
            elif 15 < next_state[1] < 600 and 1 - st_radio > 0.09:
                reward = 0


            agent.replay_buffer.push(state, action, reward, next_state, done)
            sample = agent.replay_buffer.sample(1)[0]
            agent.train(*sample)
            # agent.train()
            state = next_state
            total_reward += reward
            runtime += 1
            if done:
                break
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        print(runtime)
        if save_path is not None and (episode + 1) % 100 == 0:
            agent.save_model(save_path)
            print(f"Model saved at episode {episode + 1}")

        # 读取已存在的Excel文件或创建一个新的
        try:
            df_test = pd.read_excel('DDQN_v_data.xlsx')
        except FileNotFoundError:
            df_test = pd.DataFrame()

            # 将v_arry转换为pandas DataFrame
        df_run = pd.DataFrame({f'Test_{len(df_test.columns) + 1}': v_arry})

        # 将df_run添加到df_test中
        df_test = pd.concat([df_test, df_run], axis=1)

        # 将df_test保存到Excel文件，每次测试运行的v_arry保存在不同列中
        df_test.to_excel('DDQN_v_data.xlsx', index=False)


        try:
            df_runtime = pd.read_excel('DDQN_runtime_data.xlsx')
        except FileNotFoundError:
            df_runtime = pd.DataFrame()

            # 将runtime转换为pandas DataFrame
        column_name_runtime = f'Test_{len(df_runtime.columns) + 1}'
        df_runtime[column_name_runtime] = [runtime]

        # 将df_runtime保存到Excel文件，每次测试运行的runtime保存在不同列中
        df_runtime.to_excel('DDQN_runtime_data.xlsx', index=False)

# Testing
def test_ddqn(agent, env):
    agent.load_model("trained_q_network.pth")
    state = env.reset()
    total_reward = 0
    runtime = 0
    while True:
        velocity_space = env.get_velocity_space()
        action = agent.select_action(state, velocity_space)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        runtime += 1
        if done:
            break
    print(f"Test Complete, Total Reward: {total_reward}")
    print(f"Runtime: {runtime}")

env = RunningEnvironment()
state_size = env.state_size
action_size = len(env.get_velocity_space())
agent = DDQNAgent(state_size, action_size)

# Train
train_ddqn(agent, env, save_path="trained_q_network.pth")
# Test
test_ddqn(agent, env)
