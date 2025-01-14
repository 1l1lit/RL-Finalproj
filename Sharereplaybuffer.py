import numpy as np

# 特用于CPPO超级agent
class ReplayBuffer:
    def __init__(self, episode_length, n_rollout_threads, num_agents, act_dim, obs_dim, share_obs_dim):
        self.episode_length = episode_length
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents

        # 初始化数据存储
        self.share_obs = np.zeros((episode_length + 1, n_rollout_threads, share_obs_dim), dtype=np.float32)
        self.obs = np.zeros((episode_length + 1, n_rollout_threads, obs_dim), dtype=np.float32)
        self.actions = np.zeros((episode_length, n_rollout_threads, act_dim), dtype=np.float32)
        self.rewards = np.zeros((episode_length, n_rollout_threads, 1), dtype=np.float32)
        self.dones = np.ones((episode_length + 1, n_rollout_threads, 1), dtype=np.float32)

        self.actions_log_probs = np.zeros((episode_length, n_rollout_threads, act_dim), dtype=np.float32)
        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        self.step = 0

    def insert(self, share_obs, obs, actions, rewards, dones, value_preds, actions_log_probs):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.dones[self.step + 1] = dones.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.actions_log_probs[self.step] = actions_log_probs.copy()
        self.step = (self.step + 1) % self.episode_length

    def reset(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.actions[0] = self.actions[-1].copy()
        self.rewards[0] = self.rewards[-1].copy()
        self.dones[0] = self.dones[-1].copy()
        self.value_preds[0] = self.value_preds[-1].copy()
        self.actions_log_probs[0] = self.actions_log_probs[-1].copy()
        

    def compute_returns(self, next_value, value_normalizer, use_gae=True):

        # 计算回报 gamma = 0.99, gae_gamma = 0.95
        if not use_gae:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.returns[step + 1] * 0.99 + self.rewards[step]
        else:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                if value_normalizer != None:
                    delta = self.rewards[step] + 0.99 * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.dones[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + 0.99 * 0.95 * self.dones[step + 1] * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + 0.99 * self.value_preds[step + 1] * self.dones[step + 1] - self.value_preds[step]
                    gae = delta + 0.99 * 0.95 * self.dones[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]

    def sample(self, advantages):

        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = n_rollout_threads * episode_length

        # 随机打乱并分成mini-batches
        rand = np.random.permutation(batch_size)

        # 将数据平铺并返回
        share_obs = self.share_obs.reshape((episode_length+1)*n_rollout_threads, -1)  # 共享的观测数据
        obs = self.obs.reshape((episode_length+1)*n_rollout_threads, -1)
        actions = self.actions.reshape(episode_length*n_rollout_threads, -1)
        returns = self.returns.reshape((episode_length+1)*n_rollout_threads, -1)
        value_pred = self.value_preds.reshape((episode_length+1)*n_rollout_threads, -1)
        actions_log_probs = self.actions_log_probs.reshape(episode_length*n_rollout_threads, -1)
        advantages = advantages.reshape(episode_length*n_rollout_threads, -1)
        return (share_obs[rand], obs[rand], actions[rand], value_pred[rand], returns[rand], actions_log_probs[rand], advantages[rand])
