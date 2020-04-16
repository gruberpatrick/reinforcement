import torch
import numpy as np
from agents.ppo import PPO


ppo = PPO(3, 2)
action = ppo.get_action(
    torch.from_numpy(
        np.array([[1., 2., 3.]], dtype=np.float32)
    )
)

print(action)
