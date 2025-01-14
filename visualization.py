import torch
import numpy as np
from matplotlib import pyplot as plt


reward1 = np.array(torch.load(f"result-ippo/reward_list_999_v3.pt"))
reward2 = np.array(torch.load(f"result-cppo/reward_list_1000_v2.pt"))
reward3 = np.array(torch.load(f"result-mappo/reward_list_1000.pt"))
# reward4 = np.array(torch.load(f"result-mappo/reward_list_400_agent8_v2.pt"))

reward4 = np.array(torch.load(f"result-ippo/reward_list_400_agent2.pt"))
reward5 = np.array(torch.load(f"result-ippo/reward_list_400_agent4.pt"))
reward6 = np.array(torch.load(f"result-ippo/reward_list_400_agent8.pt"))

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,3,1)
ax.set_title("IPPO(n_agents=5)")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward per episode")
ax.plot(reward1)

bx = fig.add_subplot(1,3,2)
bx.set_title("CPPO(n_agents=5)")
bx.set_xlabel("Episode")
bx.set_ylabel("Reward per episode")
plt.plot(reward2)

cx = fig.add_subplot(1,3,3)
cx.set_title("MAPPO(n_agents=5)")
cx.set_xlabel("Episode")
cx.set_ylabel("Reward per episode")
plt.plot(reward3)

# fig = plt.figure(figsize=(15,5))
# ax = fig.add_subplot(2,2,1)
# ax.set_title("IPPO(n_agents=2)")
# ax.set_xlabel("Episode")
# ax.set_ylabel("Reward per episode")
# ax.plot(reward4)

# bx = fig.add_subplot(2,2,2)
# bx.set_title("CPPO(n_agents=4)")
# bx.set_xlabel("Episode")
# bx.set_ylabel("Reward per episode")
# plt.plot(reward5)

# cx = fig.add_subplot(2,2,3)
# cx.set_title("MAPPO(n_agents=5)")
# cx.set_xlabel("Episode")
# cx.set_ylabel("Reward per episode")
# plt.plot(reward1)

# dx = fig.add_subplot(2,2,4)
# dx.set_title("MAPPO(n_agents=8)")
# dx.set_xlabel("Episode")
# dx.set_ylabel("Reward per episode")
# plt.plot(reward6)


plt.show()