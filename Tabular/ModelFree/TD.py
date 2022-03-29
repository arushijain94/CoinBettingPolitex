import numpy as np
import itertools
"""
Implements TD approach to learn Q_hat estimate for every (s,a) pair
"""
class TD:
    """
    Get TD estimate for every Q(s,a) value of current policy
    """

    def __init__(self, env, num_samples, num_constraints, update_dual_variable):
        self.env = env
        self.S = self.env.S
        self.A = self.env.A
        self.gamma = self.env.gamma
        self.num_samples = num_samples
        self.alpha = 0.01
        self.num_times_sample = self.S * self.A * self.num_samples
        self.num_constraints = num_constraints
        self.update_dual_variable = update_dual_variable
        self.s_a_list = self.get_s_a_pairs()
        self.len_s_a_list = len(self.s_a_list)

    def get_s_a_pairs(self):
        return list(itertools.product(range(self.S), range(self.A)))

    def set_policy_prob(self, policy_prob):
        self.policy_prob = policy_prob

    def update_Q_s_a(self, s, a):
        next_s = np.random.choice(self.S, p=self.env.P[s, a])
        next_a = np.random.choice(self.A, p=self.policy_prob[next_s])
        self.Q[0, s, a] += self.alpha * (self.env.R[s, a] + self.gamma * self.Q[0, next_s, next_a] - self.Q[0, s, a])
        if self.update_dual_variable:
            self.Q[1:, s, a] += self.alpha * (
                        self.env.G[:, s, a] + self.gamma * self.Q[1:, next_s, next_a] - self.Q[1:, s, a])

    def get_Q(self):
        self.Q = np.zeros((self.num_constraints + 1, self.S, self.A))
        for _ in range(self.num_times_sample):
            s, a = self.s_a_list[np.random.choice(self.len_s_a_list)]
            self.update_Q_s_a(s, a)
        return self.Q

    def get_error_norm(self, trueQ, estimatedQ):
        """
        returns relative error which is 2 norm of difference between ||trueQ - estimatedQ||_2/||trueQ||_2
        """
        return np.linalg.norm(trueQ - estimatedQ, ord='fro') / np.linalg.norm(trueQ, ord='fro')
