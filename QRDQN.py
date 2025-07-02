import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QRDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_quantiles=51, net_arch=[128], activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_size in net_arch:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(activation_fn())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim * n_quantiles))
        self.net = nn.Sequential(*layers)
        self.n_quantiles = n_quantiles

    def forward(self, x):
        batch_size = x.size(0)
        out = self.net(x)
        return out.view(batch_size, -1, self.n_quantiles)  # [B, A, N]


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s_),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)


class QRDQN:
    def __init__(
        self,
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        gamma=0.99,
        n_quantiles=51,
        policy_kwargs=None,
        device=None,
    ):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval

        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.n_quantiles = n_quantiles

        # 默认网络结构
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[128], activation_fn=nn.ReLU)

        self.taus = torch.linspace(
            0.5 / n_quantiles, 1 - 0.5 / n_quantiles, n_quantiles
        ).view(1, n_quantiles).to(self.device)

        self.policy_net = QRDQNNetwork(
            self.state_dim,
            self.action_dim,
            n_quantiles,
            net_arch=policy_kwargs.get("net_arch", [128]),
            activation_fn=policy_kwargs.get("activation_fn", nn.ReLU),
        ).to(self.device)

        self.target_net = QRDQNNetwork(
            self.state_dim,
            self.action_dim,
            n_quantiles,
            net_arch=policy_kwargs.get("net_arch", [128]),
            activation_fn=policy_kwargs.get("activation_fn", nn.ReLU),
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)
        self.total_steps = 0
        self.total_timesteps = 0

    def _epsilon(self):
        if self.total_timesteps < self.learning_starts:
            return 1.0
        else:
            progress = (self.total_timesteps - self.learning_starts) / max(
                1.0, self.exploration_fraction * self.total_steps
            )
            return max(self.exploration_final_eps, 1.0 - progress)

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(obs).mean(dim=2)  # [1, A]
        return q_values.argmax().item()

    def _train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        for _ in range(self.gradient_steps):
            s, a, r, s_, d = self.buffer.sample(self.batch_size)
            s = torch.tensor(s, dtype=torch.float32).to(self.device)
            a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device)
            r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
            s_ = torch.tensor(s_, dtype=torch.float32).to(self.device)
            d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)

            q_dist = self.policy_net(s)  # [B, A, N]
            q_a = q_dist.gather(1, a.unsqueeze(-1).expand(-1, -1, self.n_quantiles)).squeeze(1)  # [B, N]

            with torch.no_grad():
                next_q_dist = self.target_net(s_)  # [B, A, N]
                next_q_mean = next_q_dist.mean(dim=2)
                next_a = next_q_mean.argmax(dim=1, keepdim=True)
                next_q_a = next_q_dist.gather(1, next_a.unsqueeze(-1).expand(-1, -1, self.n_quantiles)).squeeze(1)
                target = r + (1 - d) * self.gamma * next_q_a

            u = target.unsqueeze(1) - q_a.unsqueeze(2)  # [B, N, N]
            delta = 1.0
            huber = torch.where(
                u.abs() <= delta, 0.5 * u.pow(2), delta * (u.abs() - 0.5 * delta)
            )
            loss = (self.taus - (u.detach() < 0).float()).abs() * huber
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self, total_timesteps=100_000, log_interval=1000):
        obs, _ = self.env.reset()
        episode_reward = 0
        done = False
        self.total_steps = total_timesteps

        for t in range(1, total_timesteps + 1):
            self.total_timesteps = t
            eps = self._epsilon()
            if random.random() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.predict(obs)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, done or truncated)
            obs = next_obs
            episode_reward += reward

            if t > self.learning_starts and t % self.train_freq == 0:
                self._train_step()

            if t % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if done or truncated:
                obs, _ = self.env.reset()
                if t % log_interval == 0:
                    print(f"[Step {t}] Reward: {episode_reward:.2f}, ε: {eps:.3f}")
                episode_reward = 0

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
