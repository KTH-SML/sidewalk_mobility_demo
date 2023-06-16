from typing import Optional, Tuple
import casadi
import numpy as np

class GenericModel(object):
    INPUT_NOISE_STD_DEV = 0.01
    STATE_NOISE_STD_DEV = 0.01

    def __init__(self, init_state: Optional[Tuple[float]], dt: float = 0.1):
        """
        Init method for the generic model class

        :param init_state: initial state of the system
        :type init_state: Optional[Tuple[Float]]
        :param dt: simulation delta time, defaults to 0.1
        :type dt: float, optional
        """
        self.dt = dt
        # Set initial state in casadi matrix representation
        self.initial_state = casadi.DM(init_state)
        # Build model 
        self.build_model()
        # Reset trajectories
        self.reset_trajectories()

    def _build_dae(self):
        """
        Abstract method used to build the system's dae
        """
        pass

    def _build_integrator(self):
        """
        Method used to build the system's model (only ODEs are taken into account, supposing that this software
        component would work only with this kind of systems), to derive the solution of the given DAE
        """
        # Extract states from DaeBuilder and unpack iterable into tuple
        x = casadi.vertcat(*self.dae.x)
        # Extract inputs and parameters from DaeBuilder
        u = casadi.vertcat(*self.dae.u, *self.dae.p)
        # Extract system's ode
        ode = casadi.vertcat(*self.dae.ode)
        # Define DAE structure (accordingly to casadi docs)
        dae = {'x': x, 'p': u, 'ode': ode}
        options_rk = {"simplify": True, "number_of_finite_elements": 40, "tf": self.dt}
        # Create integrator
        self.integrator = casadi.integrator('integrator', 'rk', dae, options_rk)

    def _build_simulator(self):
        """
        Method used to derive a state transition function, given the integrator result
        """
        # Extract states from DaeBuilder and unpack iterable into tuple
        x = casadi.vertcat(*self.dae.x)
        # Extract inputs and parameters from DaeBuilder
        u = casadi.vertcat(*self.dae.u, *self.dae.p)
        # Evaluate integrator given current state and current input
        res = self.integrator(x0=x, p=u)
        # Extract new state for integrator result
        x_next = res['xf']
        # Create function of the form (x_t, u_t, p) -> (x_t+1)
        self._F = casadi.Function('F', [x, u], [x_next], ['x_t', 'u_t'], ['x_next'])

    def build_model(self):
        """
        Method used to build the model (daebuilder for the system's DAE, integrator for solving DAE, then simulator implementing the found solution)
        """
        # Instatiate DAE Builder
        self.dae = casadi.DaeBuilder()
        self._build_dae()

        # Instatiate empty variable for integrator
        self.integrator = None
        self._build_integrator()
        
        # Instatiate empty variable for system function
        self._F = None
        self._build_simulator()

    def set_initial_state(self, init_state):
        """
        Method used to set new system inital state

        :param init_state: initial state
        :type init_state: Tuple[float]
        """
        self.initial_state = casadi.DM(init_state)

    def reset_trajectories(self):
        """
        Method used to reset both state and input trajectories
        """
        # Reset state trajectory by resetting it to the initial state
        self.state_trajectory = [self.initial_state]
        self.input_trajectory = []

    def get_trajectories(self):
        """
        Function to return state trajectory and input trajectory (if any)

        :return: state trajectory horizontally stacked and input trajectory (if any) horizontally stacked
        :rtype: np array
        """
        if len(self.input_trajectory) > 0:
            return np.hstack(self.state_trajectory), np.hstack(self.input_trajectory)
        else:
            return np.hstack(self.state_trajectory), []
        
    def f(self, x, u, apply_input_noise=False, apply_state_noise=False):
        """
        Method used to aplly the system function 

        :param x: state of the system
        :type x: Tuple[float]
        :param u: input of the system
        :type u: Tuple[float]
        :param apply_input_noise: True applies input noise, defaults to False
        :type apply_input_noise: bool, optional
        :param apply_state_noise: True applies state noise, defaults to False
        :type apply_state_noise: bool, optional
        :return: t+1 state, control
        :rtype: Tuple[float]
        """
        # Apply input noise if any
        if apply_input_noise:
            n_inputs = len(self.dae.u)
            # Noise as Gaussian Distribution
            u[:n_inputs, 0] += np.random.normal(0, self.INPUT_NOISE_STD_DEV, n_inputs)
        # Compute x_t+1 and get it from F
        x_next = self._F(x_t=x, u_t=u)['x_next']
        # Apply state noise if any
        if apply_state_noise:
            n_states = len(self.dae.x)
            # Noise as Gaussian Distribution
            x_next += np.random.normal(0, self.STATE_NOISE_STD_DEV, n_states)
        return x_next, u
    
    def simulate(self, time_steps, u, initial_state=None, apply_input_noise=False, apply_state_noise=False):
        # Check that there is one input for the whole duration of the simulation
        assert time_steps == u.shape[1], "Input second dimension {} must match time_steps {}".format(u.shape[1], time_steps)
        # If no initial state is passed to this function, then get the already set one
        if initial_state is None:
            # Access last item of array
            initial_state = self.state_trajectory[-1]
        n_states = len(self.dae.x)
        # Istantiate result array
        res = casadi.DM.zeros((n_states, time_steps))
        # Set t+1 state to initial one, to initiate the loop
        x_next = initial_state
        for t in range(time_steps):
            # Simulate system action given current state and control
            x_next, u_applied = self.f(x_next, u[:, [t]], apply_input_noise, apply_state_noise)
            # Set new state to corresponding entry in result arrays
            res[:, [t]] = x_next
            # Append new state to state trajectory (call full() to convert n_next to array)
            self.state_trajectory.append(x_next.full())
            # Append input to input trajectory (useful if input noise is applied)
            self.input_trajectory.append(u_applied)
        # Return whole array of states
        return res.full()
