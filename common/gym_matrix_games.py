"""
Code modified from: https://github.com/alshedivat/lola/tree/master/lola
"""
# TODO change max_steps values

# from loguru import logger
from collections import deque

import numpy as np

import gym
from gym.spaces import Discrete, Tuple


class MatrixSocialDilemma(gym.Env):
    """
    A two-agent vectorized environment for matrix games.
    """

    NUM_AGENTS = 2
    VIEWPORT_W = 400
    VIEWPORT_H = 400

    def __init__(self, payout_matrix, num_actions, num_states, max_steps_per_epi=20):
        """
        :arg payout_matrix: numpy 2x2 array. Along dim 0 (rows), action of
        current agent change. Along dim 1 (col), action of the
        other agent change. (0,0) = (C,C), (1,1) = (D,D)
        :arg max_steps_per_epi: max steps per episode before done equal True
        """
        self.NUM_ACTIONS = num_actions
        self.NUM_STATES  = num_actions**2
        self.max_steps_per_epi = max_steps_per_epi
        self.payout_mat = payout_matrix
        self.action_space = Tuple([Discrete(self.NUM_ACTIONS),
                                   Discrete(self.NUM_ACTIONS)])
        self.reward_randomness = 0.0
        self.observation_space = Tuple([Discrete(self.NUM_STATES),
                                        Discrete(self.NUM_STATES)])

        self.step_count = None
        self.viewer = None
        self.observations = None

        self.epsilon = 0
        self.cc_count = deque(maxlen=max_steps_per_epi)
        self.dd_count = deque(maxlen=max_steps_per_epi)
        self.cd_count = deque(maxlen=max_steps_per_epi)
        self.dc_count = deque(maxlen=max_steps_per_epi)

    def reset(self):
        self.step_count = 0
        self.observations = (self.NUM_STATES - 1, self.NUM_STATES - 1)

        # self.observations = self._one_hot_np_arrays(self.observations, n_values=self.NUM_STATES)
        # self.observations = self._np_arrays(self.observations)

        return self.observations

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        # rewards = (self.payout_mat[ac0][ac1], self.payout_mat[ac1][ac0])
        if self.reward_randomness != 0:
            reward = float(np.random.normal(self.payout_mat[ac0][ac1], self.reward_randomness))
        else:
            reward = self.payout_mat[ac0][ac1]

        self.observations = (ac0, ac1)
        done = (self.step_count == self.max_steps_per_epi)

        # Extra log info
        self.cc_count.append(ac0 == 0 and ac1 == 0)
        self.dd_count.append(ac0 == 1 and ac1 == 1)
        self.cd_count.append(ac0 == 0 and ac1 == 1)
        self.dc_count.append(ac0 == 1 and ac1 == 0)

        self.cc_frac = (sum(list(self.cc_count)) /
                        (len(list(self.cc_count)) + self.epsilon))
        self.dd_frac = (sum(list(self.dd_count)) /
                        (len(list(self.dd_count)) + self.epsilon))
        self.cd_frac = (sum(list(self.cd_count)) /
                        (len(list(self.cd_count)) + self.epsilon))
        self.dc_frac = (sum(list(self.dc_count)) /
                        (len(list(self.dc_count)) + self.epsilon))

        info = {"extra_info_to_log": {
            "cc": self.cc_frac,
            "dd": self.dd_frac,
            "cd": self.cd_frac,
            "dc": self.dc_frac,
        }
        }

        return self.observations, reward, done, info

    def render(self, mode='human'):

        # Set windows
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W, 0, self.VIEWPORT_H)

        state_size = [[0, 0], [self.VIEWPORT_W // 2, 0],
                      [self.VIEWPORT_W // 2, self.VIEWPORT_H // 2], [0, self.VIEWPORT_H // 2]]

        # From one_hot_to_state
        # logger.debug("self.observations {}".format(self.observations))
        # state = np.nonzero(np.array(self.observations)[0])[0][0]
        state = self.observations[0]

        # logger.info("state {}".format(state))

        assert state < self.NUM_STATES and state >= 0, state
        if state == 0:
            # C & C
            delta_x = 0
            delta_y = self.VIEWPORT_H // 2
        elif state == 1:
            # C & D
            delta_x = self.VIEWPORT_W // 2
            delta_y = self.VIEWPORT_H // 2
        elif state == 2:
            # D < C
            delta_x = 0
            delta_y = 0
        elif state == 3:
            # D & D
            delta_x = self.VIEWPORT_W // 2
            delta_y = 0
        elif state == 4:
            delta_x = - self.VIEWPORT_W
            delta_y = - self.VIEWPORT_H

        current_agent_pos = state_size
        current_agent_pos = [[x + delta_x, y + delta_y] for x, y in current_agent_pos]
        self.viewer.draw_polygon(current_agent_pos, color=(0, 0, 0))
        # import time
        # time.sleep(0.5)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class IteratedMatchingPennies(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Matching Pennies game.
    """

    def __init__(self, max_steps_per_epi=20, reward_randomness=0.1):
        payout_mat = np.array([[1, -1],
                               [-1, 1]])
        super().__init__(payout_mat, max_steps_per_epi, reward_randomness)


class IteratedPrisonersDilemma(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Prisoner's Dilemma game.
    """

    def __init__(self, max_steps_per_epi=20, reward_randomness=0.1):
        payout_mat = np.array([[-1., -3],
                               [0., -2.]])
        super().__init__(payout_mat, max_steps_per_epi, reward_randomness)
        self.NAME = "IPD"


class IteratedStagHunt(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Stag Hunt game.
    """

    def __init__(self, max_steps_per_epi=20, reward_randomness=0.1):
        payout_mat = np.array([[3, 0],
                               [2, 1]])
        super().__init__(payout_mat, max_steps_per_epi, reward_randomness)


class IteratedChicken(MatrixSocialDilemma):
    """
    A two-agent vectorized environment for the Chicken game.
    """

    def __init__(self, max_steps_per_epi=20, reward_randomness=0.1):
        payout_mat = np.array([[0, -1],
                               [1, -10]])
        super().__init__(payout_mat, max_steps_per_epi, reward_randomness)

class RandomMatrixGame(MatrixSocialDilemma):

    def __init__(self, num_actions, max_steps_per_epi):
        payout_mat = np.random.rand(num_actions, num_actions) * 2 - 1
        print("Iterated Matrix Game With The Following Payout Matrix: \n", payout_mat)
        super().__init__(payout_matrix=payout_mat, num_actions=num_actions, num_states=1+num_actions**2, max_steps_per_epi=max_steps_per_epi)