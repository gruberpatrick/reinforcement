import torch
from agents.agent import Agent
from lib.memory import Memory


class Critic(torch.nn.Module):

    def __init__(self, state_size, hidden_size):

        super(Critic, self).__init__()

        self._l1 = torch.nn.Linear(state_size, hidden_size)
        self._l2 = torch.nn.Linear(hidden_size, hidden_size)
        self._l3 = torch.nn.Linear(hidden_size, 1)
        self._relu = torch.nn.ReLU()

    def forward(self, state):

        X = self._l1(state)
        X = self._relu(X)
        X = self._l2(X)
        X = self._relu(X)
        X = self._l3(X)
        return X


class Actor(torch.nn.Module):

    def __init__(self, state_size, action_size, hidden_size, categorical=True):

        super(Actor, self).__init__()

        self._categorical = categorical

        self._l1 = torch.nn.Linear(state_size, hidden_size)
        self._l2 = torch.nn.Linear(hidden_size, hidden_size)
        self._l3 = torch.nn.Linear(hidden_size, action_size)
        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(self, state):

        X = self._l1(state)
        X = self._relu(X)
        X = self._l2(X)
        X = self._relu(X)
        X = self._l3(X)

        if self._categorical:
            X = self._softmax(X)

        return X


class PPO(Agent):

    _debug = False
    _memory = Memory(fields=["state", "action", "reward", "logprob", "done"])

    def __init__(self, observation_space, action_space, hyperparams={"hidden_size": 128}):

        self._critic = Critic(observation_space, hyperparams["hidden_size"])
        self._actor = Actor(observation_space, action_space, hyperparams["hidden_size"])

    def get_action(self, state):

        # we sample to get some exploration;
        probs = self._actor(state)
        dist = torch.distributions.Categorical(probs=probs)

        if self._debug:
            print(f"get_action() probs={probs}, dist={dist}")

        action = dist.sample()
        self._last_action = action
        self._last_logprob = dist.log_prob(action)
        return action

    def train(self, state, reward, done, args):

        self._memory.add(
            state=state,
            action=self._last_action,
            reward=reward,
            logprob=self._last_logprob,
            done=done,
        )

    def end_epsiode(self):

        # policy = self._actor.forward(state)
        # estimate = self._critic.forward(state)

        pass
