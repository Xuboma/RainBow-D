import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from runner import Runner
import pandas as pd
import random


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    # 用于初始化权重和偏置的参数
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    # 重置噪声
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyDuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NoisyDuelingNetwork, self).__init__()
        # ----------输入状态处理 3——>128-------------------------
        self.fc1 = NoisyLinear(state_size, 128)
        self.fc2_adv = NoisyLinear(128, 64)

        # ----------优势函数adv(s,a)可以获得当前状态下不同动作的好坏---
        self.fc3_adv = NoisyLinear(64, action_size)

        # ----------状态价值函数V(s)可以获得当前状态的好坏------------
        self.fc2_val = NoisyLinear(128, 64)
        self.fc3_val = NoisyLinear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))

        adv = torch.relu(self.fc2_adv(x))
        adv = self.fc3_adv(adv)

        val = torch.relu(self.fc2_val(x))
        val = self.fc3_val(val)

        return val + adv - adv.mean(dim=-1, keepdim=True)


class NoisyDDQNAgent:
    def __init__(self, state_size, action_size, replay_buffer_capacity=10000, batch_size=64, alpha=0.6, beta=0.4):
        self.q_network = NoisyDuelingNetwork(state_size, action_size)
        self.target_q_network = NoisyDuelingNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = PriorityReplayBuffer(capacity=replay_buffer_capacity, alpha=alpha, beta=beta)
        self.batch_size = batch_size

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(dim=0)

        # 400 800用的
        index = action.unsqueeze(dim=-1)
        if 2 <= index <= 5:
            index = index - 2
        elif 4 <= index <= 7:
            index = index - 4
        else:
            index = index - 6
        # 1000
        # if 1 <= index <= 4:
        #     index = index - 1
        # elif 3 <= index <= 5:
        #     index = index - 3
        # else:
        #     index = index - 3

        q_values = self.q_network(state).gather(dim=-1, index=index)

        with torch.no_grad():
            next_q_values_target = self.target_q_network(next_state)

        target_actions = torch.argmax(self.q_network(next_state), dim=-1, keepdim=True)
        target_q_values = next_q_values_target.gather(dim=-1, index=target_actions)
        target_q_values = reward + gamma * (1 - done) * target_q_values

        td_error = F.smooth_l1_loss(q_values, target_q_values.detach(), reduction='none').item()

        priorities = np.abs(td_error) + 1e-5
        indices = self.replay_buffer.position - 1
        self.replay_buffer.update_priorities(indices, priorities)

        loss = F.smooth_l1_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network, self.target_q_network, 0.001)


class PriorityReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-5):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.position = 0
        self.max_priority = 1.0

    # def push(self, state, action, reward, next_state, done):
    #     transition = (state, action, reward, next_state, done)
    #     max_priority = np.max(self.priorities) if self.buffer else 1.0
    #     self.buffer.append(transition)
    #     self.priorities[self.position] = max_priority
    #     self.position = (self.position + 1) % self.capacity
    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = np.max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # If the buffer is full, overwrite the oldest transition
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        # print(len(self.buffer), len(probs))
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # print(len(self.buffer), len(probs))

        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = np.min([1., self.beta + self.beta_increment])

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        idx = indices  # 直接使用整数而不是将其放入列表
        self.priorities[idx] = priorities
        self.max_priority = max(self.max_priority, priorities)

    def __len__(self):
        return len(self.buffer)


class RunningEnvironment:
    def __init__(self):
        self.distance = 1500
        self.v = 5.5
        self.runner = Runner(60, self.distance / 1000)
        WS = self.runner.W_s(self.v)
        total_w1 = self.runner.total_w(self.distance / 1000)
        common_person_ratio = ((self.distance / self.v) * (WS+110)) / total_w1
        print('Normal person consumes {:.5%} stamina running {}m'.format(common_person_ratio, self.distance))
        self.STAMINA = total_w1
        print('Total stamina:', self.STAMINA)
        self.state_size = 2
        self.current_distance = 0
        self.remaining_endurance = 1

    def reset(self):
        self.current_distance = 0
        self.remaining_endurance = 1
        return np.array([0, self.remaining_endurance])

    # 400 800
    def get_velocity_space(self):
        if self.current_distance < self.distance*0.02:
            velocity_space = [2, 3, 4, 5]
        elif self.distance*0.02 <= self.current_distance < self.distance*0.75:
            velocity_space = [4, 5, 6, 7]
        else:
            velocity_space = [6, 7, 8, 9]
        return velocity_space
    # 1000
    # def get_velocity_space(self):
    #     if self.current_distance < self.distance*0.02:
    #         velocity_space = [1, 2, 3, 4]
    #     elif self.distance*0.02 <= self.current_distance < self.distance*0.75:
    #         velocity_space = [3, 3, 4, 5]
    #     else:
    #         velocity_space = [4, 5, 6, 7]
    #     return velocity_space

    def calculate_reward(self, sta):
        return sta * sta

    def step(self, action):
        velocity = action
        distance_covered = action
        self.current_distance += distance_covered
        WS = self.runner.W_s(velocity) + 110
        sta_radio = self.remaining_endurance - WS/self.STAMINA
        reward = self.calculate_reward(WS)
        done = self.current_distance >= self.distance
        next_state = np.array([sta_radio, self.current_distance])
        return next_state, reward, done, WS


def train_ddqn(agent, env, episodes=100, save_path=None):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        runtime = 0
        length = env.distance
        st = env.STAMINA
        v_arry = []
        for i in range(500):
            velocity_space = env.get_velocity_space()
            action = agent.select_action(state, velocity_space)
            v_arry.insert(len(v_arry), action)
            next_state, reward, done, W_S = env.step(action)
            st = st - W_S
            st_radio = st / env.STAMINA
            # print(length)
            # print(st_radio)
            # 15%
            # if next_state[1] <= length*0.02 and 1 - st_radio > 0.015:
            #     reward = -10000
            # elif 15 <= next_state[1] < length*0.75 and 1 - st_radio > 0.105:
            #     reward = -10000
            # 10%
            if next_state[1] <= length*0.1 and 1 - st_radio > 0.01:
                reward = -10000
            elif length*0.02 <= next_state[1] < length*0.86 and 1 - st_radio > 0.07:
                reward = -10000
            # 推入新的经验
            agent.replay_buffer.push(state, action, reward, next_state, done)

            for i in range(2):
                samples, indices, weights = agent.replay_buffer.sample(1)
                sample = samples[0]
                # 根据经验池中的优先级进行训练，降低误差
                agent.train(*sample)
            state = next_state
            total_reward += reward
            runtime += 1
            if done:
                break
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        print(runtime)
        if save_path is not None and (episode + 1) % 20 == 0:
            agent.save_model(save_path)
            print(f"Model saved at episode {episode + 1}")

        # 读取已存在的Excel文件或创建一个新的
        try:
            df_test = pd.read_excel('Rainbow_v_data.xlsx')
        except FileNotFoundError:
            df_test = pd.DataFrame()

            # 将v_arry转换为pandas DataFrame
        df_run = pd.DataFrame({f'Test_{len(df_test.columns) + 1}': v_arry})

        # 将df_run添加到df_test中
        df_test = pd.concat([df_test, df_run], axis=1)

        # 将df_test保存到Excel文件，每次测试运行的v_arry保存在不同列中
        df_test.to_excel('Rainbow_v_data.xlsx', index=False)


        try:
            df_runtime = pd.read_excel('Rainbow_runtime_data.xlsx')
        except FileNotFoundError:
            df_runtime = pd.DataFrame()

            # 将runtime转换为pandas DataFrame
        column_name_runtime = f'Test_{len(df_runtime.columns) + 1}'
        df_runtime[column_name_runtime] = [runtime]

        # 将df_runtime保存到Excel文件，每次测试运行的runtime保存在不同列中
        df_runtime.to_excel('Rainbow_runtime_data.xlsx', index=False)


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
agent = NoisyDDQNAgent(state_size, action_size)

# Train
train_ddqn(agent, env, save_path="trained_noisy_q_network.pth")
# Test
test_ddqn(agent, env)

# for i in range(10):
#     train_ddqn(agent, env, save_path="trained_q_network.pth")
#     # 测试
#     test_ddqn(agent, env)
