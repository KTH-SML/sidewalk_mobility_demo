import numpy as np
import casadi
from svea.models.generic_mpc import GenericModel

class SMPC(object):
    ROBOT_RADIUS = 0.2
    DELTA_TIME = 0.1
    A = 3.7
    B = 1.02
    LAMBDA = 1
    def __init__(self, model: GenericModel, x_lb, x_ub, u_lb, u_ub, n_static_obstacles, n_dynamic_obstacles, n_pedestrians, Q, R, S, N=7, apply_input_noise=False, apply_state_noise=False, verbose=False):
        """
        Init method for MPC class

        :param model: kinodynamic model of the systems
        :type model: GenericModel
        :param x_lb: state variables lower bounds
        :type x_lb: Tuple[float]
        :param x_ub: state variables upper bounds
        :type x_ub: Tuple[float]
        :param u_lb: input variables lower bounds
        :type u_lb: Tuple[float]
        :param u_ub: input variables upper bounds
        :type u_ub: Tuple[float]
        :param N: number of inputs to be predicted, defaults to 7
        :type N: int, optional
        :param apply_input_noise: True the model applies Gaussian noise the the input variables, defaults to False
        :type apply_input_noise: bool, optional
        :param apply_state_noise: True the model applies Gaussian noise the the state variables, defaults to False
        :type apply_state_noise: bool, optional
        """
        self.model = model
        # Get model's number of states and inputs
        self.n_states = len(self.model.dae.x)
        self.n_inputs = len(self.model.dae.u)

        # Get max number of static unmapped obstacles
        self.n_static_obstacles = n_static_obstacles
        # Get max number of dynamic obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        # Get max number of pedestrians
        self.n_pedestrians = n_pedestrians

        # Check that there are enough lower and upper bounds for each state/input variable
        assert self.n_states == len(x_lb), f'Number of lower bounds does not correspond to states number, number of states: {self.n_states}, number of lower bounds: {len(x_lb)}'
        assert self.n_states == len(x_ub), f'Number of lower bounds does not correspond to states number, number of states: {self.n_states}, number of lower bounds: {len(x_ub)}'
        assert self.n_inputs == len(u_lb), f'Number of lower bounds does not correspond to states number, number of states: {self.n_inputs}, number of lower bounds: {len(u_lb)}'
        assert self.n_inputs == len(u_ub), f'Number of lower bounds does not correspond to states number, number of states: {self.n_inputs}, number of lower bounds: {len(u_ub)}'
        # Get matrices of weights for cost function
        assert self.n_states == np.shape(Q)[0], f'Number of weights in states weights matrix Q does not correspond number of states, number of states: {self.n_states}, number of weights: {np.shape(Q)[0]}'
        assert self.n_inputs == np.shape(R)[0], f'Number of weights in inputs weights matrix R does not correspond number of inputs, number of inputs: {self.n_inputs}, number of weights: {np.shape(R)[0]}'
        
        # Get weights matrices
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.S = casadi.diag(S)
        # Get number of controls to be predictes 
        self.N = N
        # Get initial state
        self.initial_state = self.model.initial_state

        # Get states lower and upped bounds
        self.x_lb = casadi.DM(x_lb)
        self.x_ub = casadi.DM(x_ub)
        # Get inputs lower and upper bounds
        self.u_lb = casadi.DM(u_lb)
        self.u_ub = casadi.DM(u_ub)

        # Get noise settings
        self.apply_input_noise = apply_input_noise
        self.apply_state_noise = apply_state_noise

        # Define optimizer variables 
        self.opti = None
        self.x = None
        self.u = None
        self.reference_state = casadi.DM.zeros(self.n_states, 1)
        self.static_unmapped_obs_position = casadi.DM.zeros(2, self.n_static_obstacles)
        self.dynamic_obs_pos = casadi.DM.zeros(4, self.n_dynamic_obstacles)
        self.pedestrians_pos = casadi.DM.zeros(4, self.n_pedestrians)

        # Cost function related varirables
        self.state_diff = []
        self.angle_diff = []
        self.F_r_static = []
        self.F_r_dynamic = []
        self.F_r_sfm = []
        self.cost = 0

        # Verbose option
        self.verbose = verbose

        # Build optimizer 
        self._build_optimizer()

    def predict_position(self, obs, k):
        """
        Function to predict obstacle's next position given its pose

        :param obs: pose of the obstacles [x, y, v, yaw]
        :type obs: list[float]
        :param k: timestep
        :type k: int
        :return: new x, y coordinates
        :rtype: float, float
        """
        # Predict dynamic obs trajectory using linear motion model and estimated v, theta (dt fixed to 0.1)
        if k != 0:
            x = obs[0] + obs[2] * casadi.cos(obs[3]) * self.DELTA_TIME
            y = obs[1] + obs[2] * casadi.sin(obs[3]) * self.DELTA_TIME
        else:
            x = obs[0]
            y = obs[1]
        return x, y    

    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)  

    def _build_optimizer(self):
        """
        Method to build the optimizer problem
        """
        # Define optimizer environment
        self.opti = casadi.Opti()

        # Set optimizer variables
        self._optimizer_variables()
        # Set optimizer parameters
        self._optimizer_parameters()
        # Set optimizer cost function
        self._optimizer_cost_function()
        # Set optimizer constraints
        self._optimizer_constraints()

        # TODO: optimizer options
        p_opts = {
            "verbose": self.verbose,
            "expand": True,
            "print_in": False,
            "print_out": False,
            "print_time": False}
        s_opts = {"max_iter": 150,
                  "print_level": 1,
                  "fixed_variable_treatment": "make_constraint",
                  "barrier_tol_factor": 1,
                  }

        self.opti.solver('ipopt', p_opts, s_opts)

    def _optimizer_variables(self):
        """
        Method used to set optimizer variables
        """
        # Define optimizer variables
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(self.n_inputs, self.N)
        # Slack variable
        #self.slack = self.opti.variable(self.n_states, 1)

    def _optimizer_parameters(self):
        """
        Method used to set optimizer parameters
        """
        # Define optimizer parameters (constants)
        # Rows as the number of state variables to be kept into accout, columns how many timesteps
        self.reference_state = self.opti.parameter(self.n_states, self.N + 1)
        self.initial_state = self.opti.parameter(self.n_states, 1)
        self.static_unmapped_obs_position = self.opti.parameter(2, self.n_static_obstacles)
        self.dynamic_obs_pos = self.opti.parameter(4, self.n_dynamic_obstacles)
        self.pedestrians_pos = self.opti.parameter(4, self.n_pedestrians)

    def _optimizer_cost_function(self):
        """
        Method used to set optimizer cost function
        """
        self.cost = 0
        # Extract dimension of reference state (every state variable minus the heading)
        reference_dimension = np.shape(self.reference_state)[0] - 1
        for k in range(self.N + 1):
            # Predicted state to be as close as possible to reference one
            self.state_diff.append(self.x[:reference_dimension, k] - self.reference_state[:reference_dimension, k])
            self.cost += self.state_diff[k].T @ self.Q[:reference_dimension, :reference_dimension] @ self.state_diff[k]
            # Heading error
            self.angle_diff.append(np.pi - casadi.norm_2(casadi.norm_2(self.x[-1, k] - self.reference_state[-1, k]) - np.pi))
            self.cost += self.angle_diff[k]**2 * self.Q[-1, -1]

            # Compute obstacle repulsive force for static unmapped obstacles          
            rep_force_static = 0
            for i in range(self.n_static_obstacles):
                rep_force_static += casadi.exp(-((((self.x[0, k] - self.static_unmapped_obs_position[0, i]) ** 2  - self.ROBOT_RADIUS) / 2) + (((self.x[1, k] - self.static_unmapped_obs_position[1, i]) ** 2 - self.ROBOT_RADIUS) / 2))) / (k + 1)
            self.F_r_static.append(rep_force_static)
            self.cost += self.S[0, 0] * self.F_r_static[k]

            rep_force_dynamic = 0
            for i in range(self.n_dynamic_obstacles):
                x, y = self.predict_position(self.dynamic_obs_pos[:, i], k)
                rep_force_dynamic += casadi.exp(-((((self.x[0, k] - x) ** 2 - self.ROBOT_RADIUS) / 2) + (((self.x[1, k] - y) ** 2 - self.ROBOT_RADIUS) / 2))) / (k + 1)
            self.F_r_dynamic.append(rep_force_dynamic)
            self.cost += self.S[1, 1] * self.F_r_dynamic[k]

            sfm_x = 0
            sfm_y = 0
            sfm = 0
            x_y = casadi.MX(2, 1)
            e_p = casadi.MX(2, 1)
            v_ego = casadi.MX(2, 1)
            v_ped = casadi.MX(2, 1)
            for i in range(self.n_pedestrians):
                x_y[0], x_y[1] = self.predict_position(self.pedestrians_pos[:, i], k) 
                v_ego[0] = self.x[0, k] + self.x[2, k] * casadi.cos(self.x[3, k])
                v_ego[1] = self.x[1, k] + self.x[2, k] * casadi.sin(self.x[3, k])
                v_ped[0] = x_y[0] + self.pedestrians_pos[2, i] * casadi.cos(self.pedestrians_pos[3, i])
                v_ped[1] = x_y[1] + self.pedestrians_pos[2, i] * casadi.sin(self.pedestrians_pos[3, i])
                d_ego_p = self.x[0:2, k] - x_y
                n_ego_p = d_ego_p / casadi.norm_2(d_ego_p)
                e_p[0] = self.pedestrians_pos[2, i] * casadi.cos(self.pedestrians_pos[3, i])
                e_p[1] = self.pedestrians_pos[2, i] * casadi.sin(self.pedestrians_pos[3, i])
                y_ego_p = self.pedestrians_pos[2, i] * self.DELTA_TIME * e_p
                cos_phi_ego_p = casadi.dot(e_p, n_ego_p)
                omega = self.LAMBDA + ((1 - self.LAMBDA) * ((1 + cos_phi_ego_p) / 2))
                b_ego_p = casadi.sqrt((casadi.norm_2(d_ego_p) + casadi.norm_2(d_ego_p - ((v_ped - v_ego) * self.DELTA_TIME))) ** 2 - casadi.norm_2((v_ped - v_ego) * self.DELTA_TIME) ** 2) / 2
                g_ego_p = self.A * casadi.exp(-b_ego_p / self.B) * ((casadi.norm_2(d_ego_p) + casadi.norm_2(d_ego_p - y_ego_p)) / (2 * b_ego_p)) * 0.5 * ((d_ego_p / casadi.norm_2(d_ego_p) + ((d_ego_p - y_ego_p) / casadi.norm_2(d_ego_p - y_ego_p))))
                sfm_x += (omega * g_ego_p[0]) / (k + 1)
                sfm_y += (omega * g_ego_p[1]) / (k + 1) 
                # Narrow corridor very similar, corridor very similar, square a bit worse (maybe some tuning is needed).
                # Slowest in travel time

                #x_y[0], x_y[1] = self.predict_position(self.pedestrians_pos[:, i], k)
                #e_p[0] = self.pedestrians_pos[2, i] * casadi.cos(self.pedestrians_pos[3, i])
                #e_p[1] = self.pedestrians_pos[2, i] * casadi.sin(self.pedestrians_pos[3, i])
                #n =  (self.x[0:2, k] - self.pedestrians_pos[0:2, i]) / casadi.norm_2(self.x[0:2, k] - self.pedestrians_pos[0:2, i])
                #omega = 0.59 + (1 - 0.59) * ((1 + casadi.dot(-n, e_p)) / 2)
                #sfm += (2.66 * casadi.exp(0.65 - casadi.norm_2(self.x[0:2, k] - self.pedestrians_pos[0:2, i]) / 0.79) * omega) / (k + 1)
                # Using pedestrian preditcted positions: square worse, corridor worse, narrow corridor bit worse.
                # Overall fastest in travel time. To go back to old version, avoid using pedestrian predicted positions
            self.F_r_sfm.append(casadi.sqrt((sfm_x ** 2 + sfm_y ** 2) + 0.000001))
            #self.F_r_sfm.append(sfm)
            self.cost += self.S[2, 2] * self.F_r_sfm[k] 

            if k < self.N:
                # Weight and add to cost the control effort
                self.cost += self.u[:, k].T @ self.R @ self.u[:, k]
        # Add slack variable to the cost
        #self.cost += 1e2 * self.slack.T @ self.slack

        # Set cost function for the optimizer
        self.opti.minimize(self.cost)

    def _optimizer_constraints(self):
        """
        Method used to define optimizer constraints
        """
        # Set kinodynamic constraints given by the model for every control input to be predicted
        for t in range(self.N):
            # Generate next state given the control 
            x_next, _ = self.model.f(self.x[:, t], self.u[:, t], apply_input_noise=self.apply_input_noise, apply_state_noise=self.apply_state_noise)
            self.opti.subject_to(self.x[:, t + 1] == x_next)
            # With slack variable
            #self.opti.subject_to(self.x[:, t + 1] == x_next + self.slack)

        # Set state bounds as optimizer constraints
        self.opti.subject_to(self.opti.bounded(self.x_lb, self.x, self.x_ub))
        # Set input bounds as optimizer constraints
        self.opti.subject_to(self.opti.bounded(self.u_lb, self.u, self.u_ub))
        # Set initial state as optimizer constraint
        self.opti.subject_to(self.x[:, [0]] == self.initial_state)

    def get_ctrl(self, initial_state, reference_state, static_unmapped_obs_position, dynamic_obs_pos, pedestrian_pos):
        """
        Function to solve optimizer problem and get control from MPC, given initial state and reference state

        :param initial_state: initial state
        :type initial_state: Tuple[float]
        :param reference_state: reference state
        :type reference_state: Tuple[float]
        :return: _description_
        :rtype: _type_
        """
        # Set optimizer values for both initial state and reference states
        self.opti.set_value(self.initial_state, initial_state)
        self.opti.set_value(self.reference_state, reference_state)
        self.opti.set_value(self.static_unmapped_obs_position, static_unmapped_obs_position)
        self.opti.set_value(self.dynamic_obs_pos, dynamic_obs_pos)
        self.opti.set_value(self.pedestrians_pos, pedestrian_pos)
        
        # Solve optimizer problem if it is feasible
        try: 
            self.opti.solve()
        except RuntimeError as e:
            print(f'Cost: {self.opti.debug.value(self.cost)}')
            for angle, state, r_force_static, r_force_dynamic, r_force_sfm in zip(self.angle_diff, self.state_diff, self.F_r_static, self.F_r_dynamic, self.F_r_sfm):
                print(f'Angle diff: {self.opti.debug.value(angle)}')
                print(f'State diff: {self.opti.debug.value(state)}')
                print(f'Repulsive force static: {self.opti.debug.value(r_force_static)}')
                print(f'Repulsive force dynamic: {self.opti.debug.value(r_force_dynamic)}')
                print(f'SFM: {self.opti.debug.value(r_force_sfm)}')
            print(f'DEBUG: {self.opti.debug.value(self.test)}')
            self.opti.debug.show_infeasibilities()
            #self.opti.debug.x_describe()
            #self.opti.debug.g_describe()
            raise(e)
        if self.verbose:
            #print(f'Predicted control sequence: {self.opti.value(self.u[:, :])}')
            print(f'Cost: {self.opti.debug.value(self.cost)}')
            for angle, state, r_force_static, r_force_dynamic, r_force_sfm in zip(self.angle_diff, self.state_diff, self.F_r_static, self.F_r_dynamic, self.F_r_sfm):
                print(f'Angle diff: {self.opti.debug.value(angle)}')
                print(f'State diff: {self.opti.debug.value(state)}')
                print(f'Repulsive force static: {self.opti.debug.value(r_force_static)}')
                print(f'Repulsive force dynamic: {self.opti.debug.value(r_force_dynamic)}')
                print(f'SFM: {self.opti.debug.value(r_force_sfm)}')
        #for r_force in self.F_r_sfm:
        #    print(f'MPC Repulsive sfmc: {self.opti.debug.value(r_force)}')
        print(f'MPC Cost: {self.opti.debug.value(self.cost)}')
        # Get first control generated (not predicted ones)
        u_optimal = np.expand_dims(self.opti.value(self.u[:, 0]), axis=1)
        # Get new predicted position
        x_pred = self.opti.value(self.x)
        return u_optimal, x_pred
