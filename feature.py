from tile3 import *

class Features:
    """
    Feature class implements the one-hot encoding features
    """
    def __init__(self, num_actions):
        self.A = num_actions

    def get_one_hot_encoding(self, s, a):
        # expanded features in row format
        return int(s * self.A + a)

    def get_feature(self, s, a):
        return [self.get_one_hot_encoding(s, a)]


class TileCodingFeatures(Features):
    """
    Tile Coding class which returns the feature given (s,a) pair for tabular CMDP
    """

    def __init__(self, num_actions, iht_args):
        self.iht = IndexHashTable(**iht_args)
        self.num_features = iht_args['iht_size']
        super().__init__(num_actions)

    def get_feature(self, s, a):
        # return the tiles form tile coding function given (state,action) pair
        return self.iht.get_tiles(s, a)

    def get_feature_size(self):
        return self.num_features


class TabularTileCoding:
    """
    Tile coding arguments of Tabular environment
    """
    def __init__(self, iht_size, num_tilings, tiling_size):
        # Observation boundaries
        # (format : [[min_1, max_1], ..., [min_i, max_i], ... ] for i in state's components.
        #  state = (x, x_dot, theta, theta_dot)
        #  "Fake" bounds have been set for velocity components to ease tiling.)
        # obs_bounds = [[0,5],[0,5]] # bounds for value of state
        obs_bounds = [[0, 25]]  # bounds for value of state
        # Tiling parameters
        self._iht_args = {'iht_size': iht_size, # size of iht map
                          'num_tilings': num_tilings, # number of such grids, returns same number of non-zero 1s
                          'tiling_size': tiling_size, # constructs a [tiling_size X tiling_size] grid
                          'obs_bounds': obs_bounds}

    def get_tile_coding_args(self):
        return self._iht_args

    def get_state_representation(self, state):
        """
        Maps [0-24] state to 5X5 grid.
        """
        return [state]


class CartPoleTileCoding:
    """
    Tile coding the cartpole environment
    """
    def __init__(self, num_tilings=8, tiling_size=4):
        # Observation boundaries
        # (format : [[min_1, max_1], ..., [min_i, max_i], ... ] for i in state's components.
        #  state = (x, x_dot, theta, theta_dot)
        #  "Fake" bounds have been set for velocity components to ease tiling.)
        obs_bounds = [[-4.8, 4.8],
                      [-3., 3.],
                      [-0.25, 0.25],
                      [-3., 3.]]
        # Tiling parameters
        self._iht_args = {'iht_size': 2 ** 12,
                          'num_tilings': num_tilings,
                          'tiling_size': tiling_size,
                          'obs_bounds': obs_bounds}

    def get_tile_coding_args(self):
        return self._iht_args

