import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from runner import Runner
import random


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

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        self.buffer.append(transition)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
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


class DistributionalDuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_atoms=51):
        super(DistributionalDuelingNetwork, self).__init__()
        self.num_atoms = num_atoms

        self.fc1 = nn.Linear(state_size, 128)

        # Advantage stream
        self.fc2_adv = nn.Linear(128, 64)
        self.fc3_adv = nn.Linear(64, action_size * num_atoms)

        # Value stream
        self.fc2_val = nn.Linear(128, 64)
        self.fc3_val = nn.Linear(64, num_atoms)

    def forward(self, state):
        x = torch.relu(self.fc1(state))

        adv = torch.relu(self.fc2_adv(x))
        adv = self.fc3_adv(adv)

        val = torch.relu(self.fc2_val(x))
        val = self.fc3_val(val)

        adv = adv.view(-1, action_size, self.num_atoms)
        val = val.view(-1, 1, self.num_atoms)

        # Combine value and advantage streams
        q_values = val + adv - adv.mean(dim=1, keepdim=True)

        return F.softmax(q_values, dim=-1)


class DistributionalDDQNAgent:
    def __init__(self, state_size, action_size, num_atoms=51, replay_buffer_capacity=10000, batch_size=64, alpha=0.6, beta=0.4):
        self.q_network = DistributionalDuelingNetwork(state_size, action_size, num_atoms)
        self.target_q_network = DistributionalDuelingNetwork(state_size, action_size, num_atoms)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = PriorityReplayBuffer(capacity=replay_buffer_capacity, alpha=alpha, beta=beta)
        self.batch_size = batch_size
        self.num_atoms = num_atoms

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()  # Set the model to evaluation mode

    def select_action(self, state, velocity_space, exploration_noise=0.3):
        state = torch.tensor(state, dtype=torch.float32)

        action_values = self.q_network(state)

        max_index = torch.argmax(action_values[:, :len(velocity_space)].mean(dim=-1))

        action = velocity_space[max_index]

        return action

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, state, action, reward, next_state, done, gamma=0.99):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        action = torch.tensor([action], dtype=torch.int64)  # Ensure it's a 1D tensor
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(dim=0)

        index = action.unsqueeze(dim=-1).unsqueeze(dim=0)

        if index <= 5:
            index = index - 2
        elif 5 < index <= 6:
            index = index - 3
        else:
            index = index - 4


        q_values = self.q_network(state).gather(dim=-1, index=index)

        with torch.no_grad():
            next_q_values_target = self.target_q_network(next_state)

        target_actions = torch.argmax(self.q_network(next_state), dim=-1, keepdim=True)
        target_q_values = next_q_values_target.gather(dim=-1, index=target_actions)
        target_q_values = reward + gamma * (1 - done) * target_q_values

        td_error = F.smooth_l1_loss(q_values, target_q_values.detach(), reduction='mean').item()


        # 更新优先级
        priorities = np.abs(td_error) + 1e-5
        indices = self.replay_buffer.position - 1
        self.replay_buffer.update_priorities(indices, priorities)

        loss = F.smooth_l1_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network, self.target_q_network, 0.001)


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
        self.remaining_endurance = 100

    def reset(self):
        self.current_distance = 0
        self.remaining_endurance = 100
        return np.array([0, self.remaining_endurance])

    def get_velocity_space(self):
        if self.current_distance < 15:
            velocity_space = [2, 3, 4, 5]
        elif 15 <= self.current_distance < 600:
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
        WS = self.runner.W_s(velocity) + 110
        # 对状态进行修改加入剩余体力值
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
        for i in range(200):
            velocity_space = env.get_velocity_space()
            action = agent.select_action(state, velocity_space)
            next_state, reward, done, W_S = env.step(action)
            st = st - W_S
            st_radio = st / env.STAMINA
            if next_state[1] <= length*0.02 and 1 - st_radio > 0.015:
                reward = 0
            elif length * 0.02 <= next_state[1] < length * 0.75 and 1 - st_radio > 0.07:
                reward = 0

            agent.replay_buffer.push(state, action, reward, next_state, done)
            if i % 2 == 0:
                for _ in range(2):  # Multi-step learning, update every 2 steps
                    samples, indices, weights = agent.replay_buffer.sample(1)
                    sample = samples[0]  # 解包
                    agent.train(*sample)  # 传递解包后的参数
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
# 使用DistributionalDDQNAgent代替DDQNAgent
# agent = DDQNAgent(state_size, action_size)
agent = DistributionalDDQNAgent(state_size, action_size, num_atoms=51)

# Train
train_ddqn(agent, env, save_path="trained_q_network.pth")
# Test
test_ddqn(agent, env)
