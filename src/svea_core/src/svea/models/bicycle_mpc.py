import numpy as np
from svea.models.generic_mpc import GenericModel

class BicycleModel(GenericModel):
    TAU = 0.1 # gain for simulating SVEA's ESC
    WHEEL_BASE = 0.324  # [m] Wheelbase of SVEA vehicle
    INPUT_NOISE_STD = 0.1
    STATE_NOISE_STD = 0.01

    def __init__(self, initial_state, dt=0.1):
        """
        Init method of the BicycleModel class, calls init method of super class

        :param init_state: initial state of the system
        :type init_state: Optional[Tuple[Float]]
        :param dt: simulation delta time, defaults to 0.1
        :type dt: float, optional
        """
        super().__init__(initial_state, dt)

    def _build_dae(self):
        """
        Method used to create the system's dae
        """
        # System's variables
        # X position
        x = self.dae.add_x('x')
        # Y position
        y = self.dae.add_x('y')
        # Velocity
        v = self.dae.add_x('v')
        # Heading angle
        theta = self.dae.add_x('theta')

        # System's inputs
        # Accelearion
        a = self.dae.add_u('a')
        # Steering angle
        delta = self.dae.add_u('delta')

        # System's equations
        x_dot = v * np.cos(theta) 
        y_dot = v * np.sin(theta)
        v_dot = a
        theta_dot = v * np.tan(delta) / self.WHEEL_BASE
        # Might be also approximated to:
        #theta_dot = (v / self.WHEEL_BASE) * delta

        # Add system's equations to system's DAE
        self.dae.add_ode('x_dot', x_dot)
        self.dae.add_ode('y_dot', y_dot)
        self.dae.add_ode('v_dot', v_dot)
        self.dae.add_ode('theta_dot', theta_dot)

        # DAE's metadata (e.g. units)
        self.dae.set_unit('x', 'm')
        self.dae.set_unit('y', 'm')
        self.dae.set_unit('v', 'm/s')
        self.dae.set_unit('theta', 'rad')

if __name__ == '__main__':
    # Set initial state as x, y, v, theta
    initial_state = [0, 0, 0, 0]
    # Initialize the model
    model = BicycleModel(initial_state)
    # Simulation timesteps
    t = 20
    # Initialize empty input vector
    u = np.zeros((2, t))
    # Desired velocity
    u[0, :] = 0.5
    # Desired steering angle
    u[1, :] = 0
    # Simulate
    model.simulate(t, u)
    #model.simulate(t, u, apply_state_noise=True, apply_input_noise=True)
    x, u = model.get_trajectories()
    print(f'X trajectory: {x[0, :]}')
    print(f'Y trajectory: {x[1, :]}')
    print(f'V trajectory: {x[2, :]}')
    print(f'Theta trajectory: {x[3, :]}')
    print(f'Input velocity trajectory: {u[0, :]}')
    print(f'Input steering angle trajectory: {u[1, :]}')





