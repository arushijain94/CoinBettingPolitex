import numpy as np


class CRPO_Primal:
    """
    Paper: CRPO,[2021] (https://arxiv.org/abs/2011.05869)
    Updates the policy alternatingly between objective improvement and constraint satisfaction.
    Algorithm 1, Pg 4
    """

    def __init__(self, cmdp, num_constraints, learning_rate, eta, entropy_coeff, feature):
        self.num_constraints = num_constraints
        self.learning_rate = learning_rate
        self.eta = eta
        self.cmdp = cmdp
        self.entropy_coeff = entropy_coeff
        self.A = cmdp.A
        self.feature = feature
        self.init_policy = self.get_init_policy_prob()
        self.w_Q_until_t = []
        self.t = 0

    def update_Q_weights(self, w_Q):
        """
        w_Q: weights of Q [feature size, num_constraints+1]
        """
        self.w_Q_until_t.append(w_Q)

    def get_init_policy_prob(self):
        return np.ones(self.A) / self.A

    def increment_t_counter(self):
        self.t += 1

    def get_Q_until_time_t(self, state):
        """
        state: current state
        """
        T = len(self.w_Q_until_t)
        # print("T:", T, "len w_Q:", len(self.w_Q_until_t))
        q_until_t = np.zeros((self.num_constraints + 1, T, self.A))
        # convert to two coordinate state representation for TC
        for t in range(T):
            for a in range(self.A):
                # summing all features and then multiplying with lambda to get Q with dim [num_constraints+1]
                q_until_t[:, t, a] = np.sum(
                    np.asarray([self.w_Q_until_t[t][k] for k in self.feature.get_feature(state, a)]), axis=0)
        return q_until_t

    def check_contains_inf(self, p):
        flag_inf = False
        if np.all(np.isinf(p)):
            flag_inf = True
            return self.get_init_policy_prob(), flag_inf
        if np.any(np.isinf(p)):
            for i in range(len(p)):
                if np.isinf(p[i]):
                    p[i]=1.
                else:
                    p[i]=0.
            flag_inf = True
        return p, flag_inf

    def get_normalized_p(self, p):
        p, flag_inf = self.check_contains_inf(p)
        if flag_inf:
            return p
        p_norm = p.sum()  # 1 norm
        if p_norm == 0:
            return self.get_init_policy_prob()
        return p / p_norm

    def md_update(self, state, ind):
        """
        Mirror Ascent update for policy
        """
        if self.t == 0:
            return self.get_init_policy_prob()
        pi_t = self.get_init_policy_prob()
        q_until_t = self.get_Q_until_time_t(state)
        for t in range(self.t):
            q_t = q_until_t[ind, t]
            pi_t = np.where(pi_t < 1e-11, 1e-11, pi_t)
            q_tau_t = q_t - self.entropy_coeff * np.maximum(np.log(pi_t), -25)
            pi_t = pi_t * np.exp(np.array(self.learning_rate * q_tau_t, dtype=np.float128))
            pi_t = np.float64(pi_t)
            pi_t = self.get_normalized_p(pi_t)
        return pi_t

    def policy_prob(self, state):
        # MD update: pi_{t+1}(a|s) = pi_{t}(a|s) exp(alpha * Q(s,a))/ [\sum_{a'}pi_{t}(a'|s) exp(alpha * Q(s,a'))]
        # takes Q matrix of [ (num_constraint+1) X S X A ] dim
        # takes V_g_perf of [num_constraints] dim = \sum_{s} \rho(s) V_g(s) where \rho is initial state distribution
        if self.t == 0:
            return self.get_init_policy_prob()
        index_constraint_violation = []
        flag_constraint_violation = False
        V_c = self.get_current_v_c()
        for c in range(self.num_constraints):
            if V_c[c] < (self.cmdp.b[c] - self.eta):  # violates the constraints
                index_constraint_violation.append(c)  # records which constraints are violated
                flag_constraint_violation = True
        if not flag_constraint_violation:
            # no constraint violation then maximize Q_r objective function
            pi_t = self.md_update(state, ind=0)  # update policy in direction of Q_r
        else:
            # constraint is violated, choose randomly i \in index_constraint_violation and update policy which maximizes Q_g[i]
            violated_constraint_index = np.random.choice(index_constraint_violation)
            pi_t = self.md_update(state, ind=violated_constraint_index + 1)
        return pi_t

    def set_current_v_c(self, v_c):
        self.v_c = v_c

    def get_current_v_c(self):
        return self.v_c
