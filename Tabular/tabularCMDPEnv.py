import sys

sys.path.append("../../emdp/emdp/")
from examples import simple
import numpy as np
import scipy.optimize


class TabularCMDP:
    '''Builds a 5X5 states MDP from S&B book (2018) Example 3.5.
        Where:
        P: [S X A X T] dim transition matrix
        R: [S X A] dim reward matrix
        gamma: 0.9
        p0: uniform initial state distribution
        size: 5
        G: [S X A] dim utility matrix
        b: lower limit of utility function
        Uses gym like interface
    '''

    def __init__(self, add_constraints=True, multiple_constraints=False):
        """
        Creates a 5 X 5 CMDP,
        Parameter:
            add_constraints: False it creates a MDP; True creates a CMDP
            multiple_constraints: False adds only 1 constraint; True adds multiple constraints
        """
        self.env = simple.build_SB_example35()
        self.P = self.env.P
        self.R = self.env.R
        self.gamma = self.env.gamma
        self.p0 = self.env.p0
        self.size = self.env.size
        self.S = self.P.shape[0]
        self.A = self.P.shape[1]
        self.multiple_constraints = multiple_constraints
        self.constraints = add_constraints
        if not add_constraints:
            self.num_constraints = 0
            self.G = None
            self.b = None
        elif multiple_constraints:
            self.num_constraints = 3
            self.get_constraints(num_constraints=3)
        else:
            self.num_constraints = 1
            self.get_constraints(num_constraints=1)

    def get_constraints(self, num_constraints):
        """
        Parameter:
            num_constraints: num of constraints to add (max of 3 constraints can be added)
        """
        b = []
        G = []
        self.get_constraint_1(b, G)
        self.get_constraint_2(b, G)
        self.get_constraint_3(b, G)
        self.b = np.array(b[:num_constraints])
        self.G = np.array(G[:num_constraints])

    def get_constraint_1(self, b, G):
        """
        Constraint 1
        """
        b.append(1.5)  # lower limit
        utility_mat = np.zeros((self.S, self.A))
        utility_mat[3, :] = 1
        utility_mat[1, :] = 0.1
        G.append(utility_mat)

    def get_constraint_2(self, b, G):
        """
        Constraint 2
        """
        b.append(0.85)  # lower limit
        utility_mat = np.zeros((self.S, self.A))
        utility_mat[0, :] = 0.7
        utility_mat[4, :] = 0.2
        G.append(utility_mat)

    def get_constraint_3(self, b, G):
        """
        Constraint 3
        """
        b.append(2.7)  # lower limit
        utility_mat = np.zeros((self.S, self.A))
        utility_mat[21, :] = 0.5
        utility_mat[13, :] = 1.5
        G.append(utility_mat)

    def change_mdp(self, gamma, b):
        if gamma == 0.9:
            b = [1.5]
        elif gamma == 0.85:
            b = [1.0]
        elif gamma == 0.8:
            b = [0.7]
        elif gamma == 0.75:
            b = [0.5]
        elif gamma == 0.7:
            b = [0.38]
        self.gamma = gamma
        self.b[0] = b[0]

    def get_greedy_policy(self, qf):
        new_policy_prob = np.zeros((self.S, self.A))
        for s in range(self.S):
            max_ind = 0
            max_val = qf[s, 0]
            for a in range(1, self.A):
                if max_val < qf[s, a]:
                    max_val = qf[s, a]
                    max_ind = a
            new_policy_prob[s, max_ind] = 1
        return new_policy_prob

    def solve_dual_LP(self):
        '''
        Solve for V* & lambda*,

        min_{v,\lambda} <\rho, v> - \lambda*b
        (I - gamma * P)v >= (r + \lambda * g)

        Equivalent LP formulation:
        min_x c^T x
        -A_{ub} x >= -b_ub
        '''
        c = np.append(self.p0, -self.b)  # c = [rho_1, ..., rho_n, -b], x =[v_1, ..., v_n, lambda]
        A_ub = np.zeros((self.S * self.A, self.S + self.num_constraints))
        I = np.eye(self.S)
        extra_column = np.zeros((self.S, self.num_constraints))
        I = np.append(I, extra_column, axis=1)  # |S| X |S+num_constraints|
        for a in range(self.A):
            g = self.G[:, :, a].T
            A_ub[a * self.S:a * self.S + self.S, :] = I - np.append(self.gamma * self.P[:, a, :], g, axis=1)
        b_ub = self.R.reshape(self.S * self.A, 1, order='F')  # |S X A| X 1
        bounds = [(None, None)] * self.S
        constraints_bounds = [(0, None)] * self.num_constraints
        bounds.extend(constraints_bounds)
        res = scipy.optimize.linprog(c, A_ub=-A_ub, b_ub=-b_ub, bounds=bounds, method='simplex',
                                     options={'presolve': True, 'tol': 1e-12})
        v_opt = res.x[:-self.num_constraints]
        lambd_opt = res.x[-self.num_constraints:]
        return v_opt, lambd_opt, res.fun

    def solve_primal_LP(self, constraint=True):
        # solves primal LP of given CMDP problem
        # if constraint == True, then solve CMDP using primal LP
        # if constraint == False, solve MDP using primal LP without constraint
        '''
        Solves for mu* such that pi*(a|s) = mu*(s,a)/{\sum_a' mu*(s,a')}

        Our CMDP LP problem
        min_{mu} \sum_{s,a} -mu(s,a)r(s,a),
        s.t. \sum_{s,a} -mu(s,a)g(s,a) <= -b
        s.t. \sum_{a} mu(s,a) = (1-Y)alpha(s) + Y \sum_{s',a'}P(s|s',a') mu(s',a')

        This is equivalent to following LP formulation in Python scipy.optimize.linprog
        min_{x}  c^T .x
        s.t. A_{ub} x <= b_{ub}
        s.t. A_{eq} x = b_{eq}
        x>=0
        To use scipy library, we will reshape the 2-D state occupation measure to 1-D vector by reshape(-1)
        '''
        c = -self.R.reshape(-1)  # gives |S X A| vector
        A_eq = np.zeros((self.S, len(c)))
        for s in range(self.S):
            I = np.zeros(len(c))
            I[s * self.A:(s + 1) * self.A] = 1
            P_s = self.P[:, :, s].reshape(-1)
            A_eq[s, :] = I - self.gamma * P_s
        b_eq = self.p0
        if constraint:
            G = np.asarray(self.G)
            A_ub = G.reshape(self.num_constraints, -1)  # 3 X |S X A| matrix
            b_ub = np.asarray(self.b)
            res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=-A_ub, b_ub=-b_ub, method='simplex',
                                         options={'presolve': True, 'tol': 1e-12})
        else:
            res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='simplex',
                                         options={'presolve': True, 'tol': 1e-12})
        mu_final = res.x.reshape((self.S, self.A))
        cmdp_otimal_policy = np.zeros((self.S, self.A))
        for s in range(self.S):
            cmdp_otimal_policy[s, :] = np.divide(mu_final[s], mu_final[s].sum())
        return np.round(cmdp_otimal_policy, 2), mu_final
