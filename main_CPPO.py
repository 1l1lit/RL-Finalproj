import time
import torch
import numpy as np

from vmas import make_env
from vmas.simulator.utils import save_video
from policy import Agent
from Sharereplaybuffer import ReplayBuffer

def warmup(env, buffer):
    obs = env.reset()
    obs = [obs[f'agent_{i}'] for i in range(env.n_agents)]
    obs = np.concatenate([tensor.numpy() for tensor in obs])
    share_obs = obs.reshape(1, -1)
    buffer.share_obs[0] = share_obs.copy()

def eval_mode(agent):
        agent.actor.eval()
        agent.critic.eval()

def use_vmas_env(
    render: bool = True,
    save_render: bool = False,
    num_envs: int = 1,
    n_steps: int = 200,
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
    agent = Agent(n_agents, obs_dim=16*n_agents, share_obs_dim=16*n_agents, action_dim=2*n_agents, device=device)
    buffer = ReplayBuffer(n_steps, num_envs, n_agents, 2*n_agents, 16*n_agents, 16*n_agents)
    frame_list = []  # For creating a gif
    reward_list = []  # For plotting the rewards
    init_time = time.time()

    warmup(env, buffer)
    epochs = 10000
    version = 'v2'

    episode_rew = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        eval_mode(agent)

        # 每n_steps步训练一次
        for step in range(n_steps):

            share_obs = buffer.share_obs[step]
            action, action_prob = agent.actor(share_obs)                
            value_pred = agent.critic(share_obs)

            obs, rews, dones, info = env.step(torch.split(torch.tanh(action), action.shape[-1]//n_agents, dim=1))

            obs = [obs[f'agent_{i}'] for i in range(env.n_agents)]
            obs = np.concatenate([tensor.numpy() for tensor in obs])
            share_obs = obs.reshape(1, -1)
            dones = np.concatenate([dones for i in range(n_agents)]).reshape(-1)
            masks = np.ones((n_agents, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            rews = np.stack([rews[f'agent_{i}'] for i in range(env.n_agents)])

            buffer.insert(share_obs, share_obs, action.detach().numpy(), rews[0].reshape(1,-1), masks[0], value_pred.detach().numpy(), action_prob.detach().numpy())
            
            episode_rew += rews[0][0]

            if render and step % 20 == 0:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=visualize_render,
                )
                if save_render:
                    frame_list.append(frame)
            if dones[0]:
                env.reset()
                reward_list.append(episode_rew)
                episode_rew = 0
        
        # train
        agent.learn(buffer)
        buffer.reset()

        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
            f"episode reward: {episode_rew}"
        )
        

        if render and save_render:
            save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)
        
        if epoch % 200 == 0 and epoch != 0:
            torch.save(reward_list, f"result-cppo/reward_list_{epoch}_{version}.pt")
            torch.save(agent.actor.state_dict(), f"result-cppo/model/actor_{epoch}_{version}.pt")
            torch.save(agent.critic.state_dict(), f"result-cppo/model/critic_{epoch}_{version}.pt")
            torch.save(agent.actor_optimizer.state_dict(), f"result-cppo/model/actor_optimizer_{epoch}_{version}.pt")
            torch.save(agent.critic_optimizer.state_dict(), f"result-cppo/model/critic_optimizer_{epoch}_{version}.pt")


if __name__ == "__main__":
    use_vmas_env(
        scenario_name="balance",
        render=True,
        save_render=False,
        random_action=False,
        continuous_actions=True,
        # Environment specific
        n_agents=5,
        # max_steps=400,
    )