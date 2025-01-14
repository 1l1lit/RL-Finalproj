from critic_actor import Actor, Critic
import torch
from valuenorm import ValueNorm
from utils import *

class Agent:
    def __init__(self, n_agents, obs_dim, share_obs_dim, action_dim, gamma=0.99, clip_epsilon=0.2, lr=5e-4, \
                 critic_lr=5e-4, opti_eps=1e-5, weight_decay=0, device=torch.device("cpu"), valuenorm=True):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.critic_lr = critic_lr
        self.opti_eps = opti_eps
        self.weight_decay = weight_decay
        self.entropy_coef = 0.01
        self.device = device
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(share_obs_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.value_normalizer = ValueNorm(1) if valuenorm else None
        
    def learn(self, buffer, valuenorm=True):
        # 1. 计算回报
        next_value = self.critic(buffer.share_obs[-1])
        buffer.compute_returns(next_value.detach().numpy(), self.value_normalizer, use_gae=True)
        
        # 2. 训练
        self.actor.train()
        self.critic.train()
        # 计算优势
        if valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(10): # 训练15轮
            share_obs, obs, actions, value_pred, returns, old_action_log_probs, shuffle_adv = buffer.sample(advantages)
            value_pred = torch.from_numpy(value_pred)
            returns = torch.from_numpy(returns)
            shuffle_adv = torch.from_numpy(shuffle_adv)
            old_action_log_probs = torch.from_numpy(old_action_log_probs)
            action_log_probs, dist_entropy = self.actor.get_probs_by_action(obs, torch.from_numpy(actions))
            critic_value = self.critic(share_obs)

            # 计算 actor_loss
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * shuffle_adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * shuffle_adv
            actor_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            
            # 清空梯度
            self.actor_optimizer.zero_grad()
            (actor_loss - dist_entropy * self.entropy_coef).backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # 计算 critic_loss
            value_pred_clipped = value_pred + (critic_value - value_pred).clamp(-0.2,0.2)
            self.value_normalizer.update(returns)
            error_clipped = self.value_normalizer.normalize(returns) - value_pred_clipped
            error = self.value_normalizer.normalize(returns) - critic_value
            
            value_clipped_loss = huber_loss(error_clipped, 10.0)
            value_loss = huber_loss(error, 10.0)

            # 截断loss
            critic_loss = torch.max(value_loss, value_clipped_loss).mean()
            self.critic_optimizer.zero_grad()
            (critic_loss * 1).backward()    # critic_loss_coef=1
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()
