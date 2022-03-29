import numpy as np


class GDA_Primal:
    """
    Primal provides implementation to update the policy using OMA.
    """

    def __init__(self, env, learning_rate, feature, entropy_coeff):
        """
        Parameter:
            learning_rate: alpha
            policy: policy class object
        """
        self.t = 0
        self.learning_rate = learning_rate
        # weight matrix of Q_l until time t
        self.w_Q_until_t = []  # with dim of each element [T, num_features, num_constraints +1]
        self.A = env.A
        self.init_policy = self.get_init_policy_prob()
        self.lambd_until_t = []
        self.feature = feature
        self.entropy_coeff = entropy_coeff  # entropy coefficient

    def update_Q_weights(self, w_Q):
        """
        w_Q: weights of Q [feature size, num_constraints+1]
        """
        self.w_Q_until_t.append(w_Q)

    def get_init_policy_prob(self):
        return np.ones(self.A) / self.A

    def increment_t_counter(self):
        self.t += 1

    def update_lambda_until_t(self, current_lambd=[]):
        lambd_t = [1]
        lambd_t.extend(current_lambd)
        self.lambd_until_t.append(lambd_t)

    def get_Q_l_until_time_t(self, state):
        """
        state: current state
        """
        T = len(self.lambd_until_t)
        q_l_until_t = np.zeros((T, self.A))
        # convert to two coordinate state representation for TC
        for t in range(T):
            for a in range(self.A):
                # summing all features and then multiplying with lambda to get Q_l
                q_l_until_t[t, a] = np.dot(
                    np.sum(np.asarray(
                        [self.w_Q_until_t[t][k] for k in self.feature.get_feature(state, a)]), axis=0),
                    self.lambd_until_t[t])
        return q_l_until_t

    def policy_prob(self, state):
        """
        policy update -> pi_{t+1} = pi_{t} exp(alpha * (Q - \tau \log \pi_t))
        Parameter:
            Q: [S x A] dim state-action value function
        """
        if self.t == 0:
            return self.init_policy
        pi_t = self.get_init_policy_prob()
        q_l_t = self.get_Q_l_until_time_t(state)
        for t in range(self.t):
            pi_t = np.where(pi_t < 1e-11, 1e-11, pi_t)
            q_tau_t = q_l_t[t] - self.entropy_coeff * np.maximum(np.log(pi_t), -25)
            pi_t = pi_t * np.exp(np.array(self.learning_rate * q_tau_t, dtype=np.float128))
            pi_t = np.float64(pi_t)
            pi_t = self.get_normalized_p(pi_t)
        return pi_t

    def check_contains_inf(self, p):
        flag_inf =  False
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


class GDA_Dual:
    """
    Dual: Provides implementation to update dual variable lambda using Gradient Descent.
    """

    def __init__(self, learning_rate, b, lower_limit, upper_limit, num_constraints):
        """
        Parameter:
          b: constraint threshold
          lower_limit: lower limit for projection
          upper_limit: upper limit for projection
          learning_rate: learning rate for dual update
          lambd: initial value of dual variable lambd
        """
        self.b = b
        self.learning_rate = learning_rate
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.curr_lambd = np.ones(num_constraints)  # initial lambd value

    def get_projected_value(self, x):
        return np.maximum(self.lower_limit, np.minimum(x, self.upper_limit))

    def update(self, value_utility):
        """
        lambda update -> lambda_{t+1} = P_[lower, upper](lambda_{t} - alpha*(value_utility - b))
        """
        self.curr_lambd = self.get_projected_value(self.curr_lambd - self.learning_rate * (value_utility - self.b))
