import numpy as np

class Policy:
    """
    Creates a policy with initialization to a random policy
    """

    def __init__(self, env, rng):
        self.env = env
        S = self.env.S
        A = self.env.A
        self.policy_prob = rng.dirichlet(np.ones(A), size=S)  # initialize with random prob

    def set_policy_prob(self, policy_prob):
        # takes [S X A] dim policy and initialize with given policy as in argument
        self.policy_prob = policy_prob

    def get_V_function(self, reward):
        # computes the V^{\pi} = (I - \gamma* P_\pi)^{-1} R_\pi
        # returns  a [S] dimensional vector
        P = self.env.P
        discount = self.env.gamma
        ppi = np.einsum('sat,sa->st', P, self.policy_prob)
        rpi = np.einsum('sa,sa->s', reward, self.policy_prob)
        vf, _, _, _ = np.linalg.lstsq(np.eye(P.shape[-1]) - discount * ppi, rpi,
                                      rcond=None)  # to resolve singular matrix issues, used least square method rather than solve
        return vf

    def get_Q_function(self, reward):
        # computes the Q^{\pi} = R + (\gamma * P * V_\pi)
        # returns a [S x A] array
        P = self.env.P
        discount = self.env.gamma
        vf = self.get_V_function(reward)
        Qf = reward + (discount * np.einsum('sat,t->sa', P, vf))
        return Qf

