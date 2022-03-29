from Cartpole import *


class CB_Primal:
    """
    Paper: Coin Betting and Parameter-Free Online Learning, Orabona and Pal [2016] (https://arxiv.org/abs/1602.04128)
    Creates a CB algorithm to update the simplex policy using Learning with Expert Advice (LEA) based on KT potential,
    Algorithm 2, Pg 8
    """

    def __init__(self, env, T, feature, num_constraints, discount_factor, entropy_coeff, ul_lambd=0.0):
        self.t = 0
        self.A = env.A
        self.num_iter = T
        self.num_constraints = num_constraints
        self.feature = feature
        # weight matrix of Q_l until time t
        self.w_Q_until_t = []  # with dim of each element [T, num_features, num_constraints +1]
        # lambda vector until time t
        self.lambd_until_t = []
        self.entropy_coeff = entropy_coeff  # entropy coefficient
        self.init_policy = self.get_init_policy()
        self.grad_max_policy = self.get_max_grad_policy(discount_factor, ul_lambd)

    def get_init_policy(self):
        # returns a uniform policy
        return np.ones(self.A) / self.A

    def update_Q_weights(self, w_Q):
        """
        w_Q: weights of Q [feature size, num_constraints+1]
        """
        self.w_Q_until_t.append(w_Q)

    def get_max_grad_policy(self, gamma, ul_lambd):
        # To normalize the gradient of policy, value of max of grad of policy
        # Note: Assumption here is that 0<= reward, cost < =1.
        # If assumption change then multiply with max reward/cost
        return (1 + np.sum(ul_lambd)) / (1 - gamma) + self.entropy_coeff * 25

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
                q_l_until_t[t, a] = np.dot(np.sum(np.asarray(
                    [self.w_Q_until_t[t][k] for k in self.feature.get_feature(state, a)]), axis=0),
                    self.lambd_until_t[t])
        return q_l_until_t

    def increment_t_counter(self):
        self.t += 1

    def policy_prob(self, state):
        """
        Use Coin betting simplex algorithm to get policy at a current iteration
        Parameter:
            state: state
        Returns:
            \pi_t(s): [num of actions] prob vector
        """
        if self.t == 0:
            return self.get_init_policy()
        # get q_l vector until time t which is normalized
        q_l_t = self.get_Q_l_until_time_t(state)
        # maintains sum of a*w until time t
        sum_a_w = np.ones(self.A)
        # maintains sum of a until time t
        sum_a_t = np.zeros(self.A)
        # w value at time t
        w_t = np.ones(self.A)
        # current pi_t
        pi_t = self.get_init_policy()
        for t in range(self.t):
            # get Q_\tau = Q_l - \tau* \log pi_t
            pi_t = np.where(pi_t < 1e-11, 1e-11, pi_t)
            q_tau_t = q_l_t[t] - self.entropy_coeff * np.maximum(np.log(pi_t), -25)
            q_tau_t /= self.grad_max_policy
            # advantage a at time t
            a_t = self.get_advantage_val(q_tau_t, pi_t)
            for a in range(self.A):
                if w_t[a] <= 0:
                    a_t[a] = max(0, a_t[a])
            sum_a_t += a_t
            sum_a_w += np.multiply(a_t, w_t)
            w_t = np.multiply(sum_a_t, sum_a_w) / (t + 1)  # element-wise product
            p_t = np.multiply(self.init_policy, np.maximum(0, w_t))
            pi_t = self.get_normalized_p(p_t)
        return pi_t

    def get_advantage_val(self, q, action_prob):
        # Q(s,a) - V(s)
        return q - np.dot(q, action_prob)

    def get_normalized_p(self, p):
        p_norm = p.sum()  # 1 norm
        if p_norm == 0:
            return self.init_policy
        else:
            return p / p_norm


class CB_Dual:
    """
    Paper: Training Deep Networks without Learning Rates Through Coin Betting,
     Orabona [2016] (https://arxiv.org/abs/1705.07795)
    Creates a CB algorithm to update the dual variable \lambda using COCOB-backprop, Algorithm 2, Pg 6
    """

    def __init__(self, num_constraints, alpha, lower_limit_lambd, upper_limit_lambd):
        self.lower_limit_lambd = lower_limit_lambd
        self.upper_limit_lambd = upper_limit_lambd
        self.num_constraints = num_constraints
        self.curr_lambd = self.get_projected_lambd(np.ones(num_constraints))  # lambda initialised to 1
        self.initial_lambd = self.get_projected_lambd(np.ones(num_constraints))  # lambda initialised to 1
        self.alpha_lambd = alpha
        self.G_lambd = np.zeros(num_constraints)  # G_t = G_{t-1} + |g_t|, where G_0 = L
        self.L_lambd = np.zeros(num_constraints)  # L_t = max(L_{t−1}, |g_t|)
        self.reward_lambd = np.zeros(num_constraints)  # Reward_t = Reward_{t-1} + (w_t - w_1)g_t, where R_0 = 0
        self.theta_lambd = np.zeros(num_constraints)  # theta_{t} = theta_{t-1} + g_t, where theta_0 = 0
        self.w_lambd = np.ones(num_constraints)  # w_t = w_1 + \beta(L + Reward_t), where w_1 = projected(1)

    def update(self, g):
        """
        Implements COCOB Algorithm 2 for updating lambda variable a d-dim vector
        Parameter:
            g: negative value of un-normalized gradient of the regret for dual variable
            Here,
            R(\lambda^*, T) = \sum_t <\lambda_t - \lambda^*, (\hat V_c - b)>,
            where V_c is cost value function and b is threshold for cost constraint.
        """
        self.L_lambd = np.maximum(self.L_lambd, np.abs(g))
        self.G_lambd += np.abs(g)
        self.reward_lambd = np.maximum(self.reward_lambd + (self.w_lambd - self.initial_lambd) * g, 0)
        self.theta_lambd += g
        beta = self.theta_lambd / (
                self.L_lambd * np.maximum(self.G_lambd + self.L_lambd, self.alpha_lambd * self.L_lambd))
        self.curr_lambd = self.get_projected_lambd(self.initial_lambd + beta * (self.L_lambd + self.reward_lambd))

    def get_projected_lambd(self, lambd):
        """
        Parameter:
            lambd
        Returns:
            projected value of lambd
        Here, lambd \in [lower_limit_lambd, upper_limit_lambd]
        """
        return np.maximum(self.lower_limit_lambd, np.minimum(self.upper_limit_lambd, lambd))
