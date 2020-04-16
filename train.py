import argparse
import gym
import json
import logging
import neptune
import pickle
import sys

import agents.q_learning

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


class Main:

    _env = None

    def __init__(self, args):

        # JSON decode the agent's hyperparameters and save them;
        args.hyperparams = json.loads(args.hyperparams)
        self._args = args

        self._console_log = args.console_log
        self._neptune_log = args.neptune_log

        if self._neptune_log:
            neptune.init("gruberpatrick/game-ai")
            self._experiment = neptune.create_experiment(
                upload_source_files=[
                    "./agents/agent.py",
                    agents.q_learning.__file__,
                ],
                params=self._args.hyperparams,
            )

        self._env = gym.make(self._args.environment)
        self._agent = self.get_agent(self._args.agent)(
            self._env.observation_space,
            self._env.action_space,
            hyperparams=self._args.hyperparams,
        )

    def get_agent(self, agent_name):

        if agent_name == "q_learning":
            return agents.q_learning.QLearning

        raise NotImplementedError(f"Agent named {agent_name} does not exist.")

    def reset_env(self):

        self._total_reward = 0
        self._steps = 0
        return self._env.reset()

    def episode(self):

        state = self.reset_env()
        done = False

        while not done:
            action = self._agent.get_action(state)
            state, reward, done, args = self._env.step(action)
            self._total_reward += reward
            self._steps += 1
            self._agent.train(state, reward, done, args)

        self._agent.end_epsiode()

    def logging(self):

        agent_logs = self._agent.get_logging()

        # create the logging string and write to the console;
        if self._console_log:
            agent_logging = f"Episode finished: steps={self._steps}, reward={self._total_reward}, "
            for key in agent_logs:
                agent_logging += f"{key}:{agent_logs[key]}, "
            log.info(agent_logging)

        # write the logging to neptune;
        if self._neptune_log:
            neptune.log_metric("steps", self._steps)
            neptune.log_metric("reward", self._total_reward)
            for key in agent_logs:
                neptune.log_metric(key, agent_logs[key])

    def save_experiment(self):

        with open(f"./output/{self._experiment._id}.ckp", "wb") as fh:
            pickle.dump(self._agent, fh)
        neptune.log_artifact(f"./output/{self._experiment._id}.ckp")
        neptune.append_tag(self._args.agent)
        neptune.append_tag(self._args.environment)

    def run(self):

        # run through predefined amount of episodes;
        for _ in range(self._args.episodes):
            self.episode()
            self.logging()

        # save the experiment tp disk and neptune;
        self.save_experiment()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reinforcement learning trainer.")

    parser.add_argument(
        "--episodes",
        action="store",
        default=5000,
        type=int,
        required=False,
        help="(default=5000) The amount of episodes to train for.",
    )

    parser.add_argument(
        "--agent",
        action="store",
        type=str,
        choices=("q_learning", ""),
        required=True,
        help="The name of the agent to be used.",
    )

    parser.add_argument(
        "--environment",
        action="store",
        type=str,
        required=True,
        help="The environment to run the agent against.",
    )

    parser.add_argument(
        "--hyperparams",
        action="store",
        default="{}",
        type=str,
        required=False,
        help="(default={}) JSON encoded hyperparameters, passed to the agent.",
    )

    parser.add_argument(
        '--console_log',
        action='store_true',
        help='Log training stats to console.',
    )

    parser.add_argument(
        '--neptune_log',
        action='store_false',
        help='Log training stats to neptune.ai.',
    )

    main = Main(parser.parse_args())
    main.run()
