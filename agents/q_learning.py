import numpy as np

from agents.agent import Agent
from gym import spaces


class QLearning(Agent):

    def __init__(self, observation_space, action_space, hyperparams={}):

        # handle the observation and action space types;
        if not isinstance(observation_space, spaces.Discrete):
            raise Exception("The environment observation space needs to be discrete.")
        if not isinstance(action_space, spaces.Discrete):
            raise Exception("The environment action space needs to be discrete.")

        # get call parameters;
        self._observation_space = observation_space.n
        self._action_space = action_space.n
        self._q = np.zeros((self._observation_space, self._action_space))
        self._hyperparams = hyperparams

        # load hyperparams;
        self._alpha = self._hyperparams.get("alpha", .5)
        self._gamma = self._hyperparams.get("gamma", .9)
        self._epsilon = self._hyperparams.get("epsilon", 1)
        self._epsilon_decay = self._hyperparams.get("epsilon_decay", .98)
        self._epsilon_min = self._hyperparams.get("epsilon_min", 0)

    def get_action(self, state):

        if self._epsilon > np.random.rand():
            action = np.random.randint(self._action_space)
        else:
            action = np.argmax(self._q[state, :])
        self._last_state = state
        self._last_action = action
        return action

    def train(self, state, reward, done, args):

        last_value = self._q[self._last_state, self._last_action]

        self._q[self._last_state, self._last_action] = (
            (1 - self._alpha) * last_value
            + self._alpha * (reward + self._gamma * np.max(self._q[state, :]) - last_value)
        )

    def end_epsiode(self):

        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def get_logging(self):

        return {
            "q_learning_epsilon": self._epsilon,
        }
