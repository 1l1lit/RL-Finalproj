import time
import torch
import numpy as np

from vmas import make_env
from vmas.simulator.utils import save_video
from policy import Agent
from replaybuffer import ReplayBuffer

def warmup(env, buffers):
    obs = env.reset()
    obs = [obs[f'agent_{i}'] for i in range(env.n_agents)]
    obs = np.concatenate([tensor.numpy() for tensor in obs])
    share_obs = obs.reshape(1, -1)
    for i in range(env.n_agents):
        buffers[i].share_obs[0] = share_obs.copy()
        buffers[i].obs[0] = obs[i].copy()

def eval_mode(agents):
    for i in range(len(agents)):
        agents[i].actor.eval()
        agents[i].critic.eval()

def use_vmas_env(
    render: bool = True,
    save_render: bool = False,
    num_envs: int = 1,
    n_steps: int = 200, # default 200
    random_action: bool = False,
    device: str = "cpu",
    scenario_name: str = "balance",
    continuous_actions: bool = True,
    visualize_render: bool = True,
    dict_spaces: bool = True,
    **kwargs,
):

    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        dict_spaces=dict_spaces,
        wrapper=None,
        seed=None,
        # Environment specific variables
        **kwargs,
    )

    n_agents = kwargs.get('n_agents', None)
    agents = [Agent(n_agents, obs_dim=16, share_obs_dim=16*n_agents, action_dim=2, device=device) for _ in range(n_agents)]
    buffers = [ReplayBuffer(n_steps, num_envs, n_agents, 2, 16, 16*n_agents) for _ in range(n_agents)]
    frame_list = []  # For creating a gif
    reward_list = []  # For plotting the rewards
    init_time = time.time()

    warmup(env, buffers)
    epochs = 1000
    version = 'agent2'

    episode_rew = 0
    for epoch in range(epochs):

        print(f"Epoch: {epoch}")
        eval_mode(agents)

        # 每n_steps步训练一次
        for step in range(n_steps):
            
            actions = []
            value_preds = []
            action_probs = []
            for i in range(n_agents):
                obs = buffers[i].obs[step]
                share_obs = buffers[i].share_obs[step]
                action, action_prob = agents[i].actor(obs)                
                value_pred = agents[i].critic(share_obs)
                actions.append(action)
                action_probs.append(action_prob)
                value_preds.append(value_pred)

            obs, rews, dones, info = env.step([torch.tanh(actions[i]) for i in range(n_agents)])

            obs = [obs[f'agent_{i}'] for i in range(env.n_agents)]
            obs = np.concatenate([tensor.numpy() for tensor in obs])
            share_obs = obs.reshape(1, -1)
            dones = np.concatenate([dones for i in range(n_agents)]).reshape(-1)
            masks = np.ones((n_agents, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            rews = np.stack([rews[f'agent_{i}'] for i in range(env.n_agents)])

            for i in range(n_agents):
                buffers[i].insert(share_obs, obs[i], actions[i].detach().numpy(), rews[i].reshape(1,-1), masks[i], value_preds[i].detach().numpy(), action_probs[i].detach().numpy())

            if render and step % 20 == 0:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=visualize_render,
                )
                if save_render:
                    frame_list.append(frame)

            episode_rew += rews[0][0]
            if dones[0]:
                env.reset()
                reward_list.append(episode_rew)
                episode_rew = 0

        # train
        for i in range(n_agents):
            agents[i].learn(buffers[i])
            buffers[i].reset()

        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
            f"total reward: {episode_rew}"
        )

        if render and save_render:
            save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)
        
        if (epoch % 200 == 0 and epoch != 0) or epoch == epochs - 1:
            torch.save(reward_list, f"result-mappo/reward_list_{epoch}_{version}.pt")
            for i in range(n_agents):
                torch.save(agents[i].actor.state_dict(), f"result-mappo/model/actor_{i}_{epoch}_{version}.pt")
                torch.save(agents[i].critic.state_dict(), f"result-mappo/model/critic_{i}_{epoch}_{version}.pt")
                torch.save(agents[i].actor_optimizer.state_dict(), f"result-mappo/model/actor_optimizer_{i}_{epoch}_{version}.pt")
                torch.save(agents[i].critic_optimizer.state_dict(), f"result-mappo/model/critic_optimizer_{i}_{epoch}_{version}.pt")


if __name__ == "__main__":
    use_vmas_env(
        scenario_name="balance",
        render=True,
        save_render=False,
        random_action=False,
        continuous_actions=True,
        # Environment specific
        n_agents=2,
        # max_steps=400,
    )