"""
15th version of AB predator-prey simulation

Characteristics:
- Two species (prey; sheep, predator; wolves)
    * same metabolism
    * same speed parameters
    + wolves only catch the closest sheep
- No resource; no grass patches
- Agents move randomly (both linear and angular)
- No Energy/Mass/Speed trade-offs: fixed speed, fixed metabolic cost
- No reproduction and death dynamics
- Storing separate files for render data

- No grass patches
- Instead; Energy distribution based on local sheep density

- remove fitness (only use energy)
- extract energy at each ts


- CMAES
"""
import os

from evosax.algorithms import CMA_ES

from structs import *
from functions import *

import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
from sensor import jit_get_all_agent_sensors


WORLD_SIZE_X = 400.0
WORLD_SIZE_Y = 400.0

MAX_SPAWN_X = 300.0
MAX_SPAWN_Y = 300.0

Dt = 0.1 # discrete time increments
KEY = jax.random.PRNGKey(0)
NOISE_SCALE = 0.05
DAMPING = 0.1

# Sheep parameters
NUM_SHEEP = 10
SHEEP_RADIUS = 5.0
SHEEP_ENERGY_BEGIN_MAX = 50.0
SHEEP_MASS_BEGIN = 5.0 # initial sheep mass at birth
SHEEP_AGENT_TYPE = 1
EAT_RATE_SHEEP = 0.2

# Wolf parameters
NUM_WOLF = 10
WOLF_RADIUS = 7.0
WOLF_ENERGY_BEGIN_MAX = 50.0
WOLF_MASS_BEGIN = 7.0
WOLF_AGENT_TYPE = 2

# Metabolism
METABOLIC_COST_SPEED = 0.01
METABOLIC_COST_ANGULAR = 0.05
BASIC_METABOLIC_COST_SHEEP = 0.02
BASIC_METABOLIC_COST_WOLF = 0.04

# Energy parameters (replaces grass)
BASE_ENERGY_RATE = 0.05  # base energy gain for sheep per timestep

# Action parameters (sheep)
SHEEP_SPEED_MULTIPLIER = 2.0
SHEEP_LINEAR_ACTION_SCALE = SHEEP_SPEED_MULTIPLIER * SHEEP_RADIUS / Dt
SHEEP_ANGULAR_SPEED_SCALE = 5.0

# Action parameters (wolves)
WOLF_SPEED_MULTIPLIER = 2.5
WOLF_LINEAR_ACTION_SCALE = WOLF_SPEED_MULTIPLIER * WOLF_RADIUS / Dt
WOLF_ANGULAR_SPEED_SCALE = 5.0

# Sensors parameters
SHEEP_RAY_MAX_LENGTH = 120.0
WOLF_RAY_MAX_LENGTH = 120.0
RAY_RESOLUTION = 9  # W&B update
RAY_SPAN = jnp.pi/3 # W&B update

# Controller parameters
NUM_OBS = RAY_RESOLUTION*4 + 5
NUM_NEURONS = 100
NUM_ACTIONS = 2
ACTION_SCALE = 1.0
LINEAR_ACTION_OFFSET = 0.0
TIME_CONSTANT_SCALE = 10.0 # speed of the neuron dynamics
NUM_ES_PARAMS = NUM_NEURONS * (NUM_NEURONS + NUM_OBS + NUM_ACTIONS + 2) # total number of parameters the Evolutionary Strategy needs to optimize


# Training parameters
NUM_WORLDS = 1
NUM_GENERATIONS = 1
EP_LEN = 100
#ELITE_RATIO = 0.3
#SIGMA_INIT = 0.1
ENERGY_THRESH_SAVE = 150.0
ENERGY_THRESH_SAVE_STEP = 10.0

#FITNESS_THRESH_SAVE = 150.0 # threshold for saving render data
#FITNESS_THRESH_SAVE_STEP = 10.0 # the amount by which we increase the threshold for saving render data

# save data
DATA_PATH = "./data/sheep_wolf_data5/"


# Predator-prey world parameters
PP_WORLD_PARAMS = Params(content= {"sheep_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_begin_max": SHEEP_ENERGY_BEGIN_MAX,
                                                    "mass_begin": SHEEP_MASS_BEGIN,
                                                    "eat_rate": EAT_RATE_SHEEP,
                                                    "radius": SHEEP_RADIUS,
                                                    "agent_type": SHEEP_AGENT_TYPE,
                                                    "num_sheep": NUM_SHEEP
                                                    },
                                   "wolf_params": {"x_max": MAX_SPAWN_X,
                                                   "y_max": MAX_SPAWN_Y,
                                                   "energy_begin_max": WOLF_ENERGY_BEGIN_MAX,
                                                   "mass_begin": WOLF_MASS_BEGIN,
                                                   "radius": WOLF_RADIUS,
                                                   "agent_type": WOLF_AGENT_TYPE,
                                                   "num_wolf": NUM_WOLF
                                                   },
                                   "policy_params_sheep": {"num_neurons": NUM_NEURONS,
                                                     "num_obs": NUM_OBS,
                                                     "num_actions": NUM_ACTIONS
                                                    },
                                   "policy_params_wolf": {"num_neurons": NUM_NEURONS,
                                                     "num_obs": NUM_OBS,
                                                     "num_actions": NUM_ACTIONS
                                                    }})


# Sheep dataclass
@struct.dataclass
class Sheep(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        policy = params.content["policy"]

        key, *subkeys = random.split(key, 5)

        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_begin_max = params.content["energy_begin_max"]
        eat_rate = params.content["eat_rate"]
        radius = params.content["radius"]
        mass_begin = params.content["mass_begin"]  # initial mass

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max, "energy_begin_max": energy_begin_max, "mass": mass_begin, "eat_rate": eat_rate}
        params = Params(content=params_content)

        def create_active_agent():
            x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max) # random initial position (x,y)
            y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
            ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)  # random initial angle
            x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize x velocity as 0
            y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize y velocity as 0
            ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize angular velocity as 0

            energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([-1.0]), "y": jnp.array([-1.0]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([-1.0])} # placeholder values
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)

        return Sheep(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=policy)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            # input
            obs_rays = input.content["obs"]
            energy_intake = input.content["energy_intake"] # also handles energy output (if eaten by wolves)

            # current agent state
            energy = agent.state.content["energy"]
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"] # current x_velocity
            y_dot = agent.state.content["y_dot"] # current y_velocity
            ang_dot = agent.state.content["ang_dot"]

            obs_content = {'obs': jnp.concatenate((obs_rays,
                                                  energy,
                                                  jnp.array([energy_intake]).reshape(1),
                                                  x_dot, y_dot, ang_dot), axis=0)}

            obs = Signal(content=obs_content)

            # add: new policy - use obs in new policy
            new_policy = CTRNN.step_policy(agent.policy, obs, step_params)

            dt = step_params.content["dt"]
            damping = step_params.content["damping"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            #eat_rate = agent.params.content["eat_rate"]

            action = new_policy.state.content["action"]
            forward_action = action[0]  # sigmoid (0 to 1)
            angular_action = action[1]  # tanh (-1 to 1)

            key, *noise_keys = random.split(agent.key, 3)

            # fixed base speed (with noise)
            speed = ((LINEAR_ACTION_OFFSET + SHEEP_LINEAR_ACTION_SCALE * forward_action) *
                     (1 + NOISE_SCALE * jax.random.normal(noise_keys[0], ())))
            ang_speed = SHEEP_ANGULAR_SPEED_SCALE * angular_action * (1 + NOISE_SCALE * jax.random.normal(noise_keys[1], ()))

            # updated positions
            #x_new = jnp.clip(x + dt * x_dot, -x_max_arena, x_max_arena)  # no wrap-around for now; may change it later
            #y_new = jnp.clip(y + dt * y_dot, -y_max_arena, y_max_arena)

            # wraparound
            x_new = jnp.mod(x + dt * x_dot + x_max_arena, 2 * x_max_arena) - x_max_arena
            y_new = jnp.mod(y + dt * y_dot + y_max_arena, 2 * y_max_arena) - y_max_arena
            ang_new = jnp.mod(ang + dt * ang_dot + jnp.pi, 2 * jnp.pi) - jnp.pi

            x_dot_new = speed * jnp.cos(ang) - dt * x_dot * damping
            y_dot_new = speed * jnp.sin(ang) - dt * y_dot * damping
            ang_dot_new = ang_speed - dt * ang_dot * damping

            # fixed metabolic cost
            metabolic_cost = BASIC_METABOLIC_COST_SHEEP
            energy_new = energy + energy_intake - metabolic_cost # energy_intake already includes loss to wolves

            agent_is_dead = energy_new[0] <= 0.0

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new, "ang_dot": ang_dot_new,
                                 "energy": energy_new}
            new_state = State(content=new_state_content)

            return jax.lax.cond(
                agent_is_dead,
                lambda _: agent.replace(state=new_state, active_state=0),  # mark as dead/inactive
                lambda _: agent.replace(state=new_state, key=key, age=agent.age + dt, policy=new_policy), # add new policy
                None
            )
        def step_inactive_agent():
            return agent

        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)

    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]
        eat_rate = agent.params.content["eat_rate"]
        key = agent.key

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key)


@struct.dataclass
class Wolf(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        policy = params.content["policy"]
        key, *subkeys = random.split(key, 5)

        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_begin_max = params.content["energy_begin_max"]
        radius = params.content["radius"]
        mass_begin = params.content["mass_begin"]

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max, "energy_begin_max": energy_begin_max, "mass": mass_begin}
        params = Params(content=params_content)

        def create_active_agent():
            x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)  # random initial position (x,y)
            y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
            ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)  # random initial angle
            x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize x velocity as 0
            y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize y velocity as 0
            ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # initialize angular velocity as 0

            energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                             "energy": energy}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([-1.0]), "y": jnp.array([-1.0]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([-1.0])}
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)
        return Wolf(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=policy)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            energy_intake = input.content["energy_intake"]
            obs_rays = input.content["obs"]

            energy = agent.state.content["energy"]
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"]  # current x_velocity
            y_dot = agent.state.content["y_dot"]  # current y_velocity
            ang_dot = agent.state.content["ang_dot"]

            obs_content = {'obs': jnp.concatenate((obs_rays,
                                                   energy,
                                                   jnp.array([energy_intake]).reshape(1),
                                                   x_dot, y_dot, ang_dot), axis=0)}
            obs = Signal(content=obs_content)

            new_policy = CTRNN.step_policy(agent.policy, obs, step_params)

            dt = step_params.content["dt"]
            damping = step_params.content["damping"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            action = new_policy.state.content["action"]
            forward_action = action[0]  # sigmoid (0 to 1)
            angular_action = action[1]  # tanh (-1 to 1)

            key, *noise_keys = random.split(agent.key, 3)

            # fixed base speed (with noise)
            speed = (LINEAR_ACTION_OFFSET + WOLF_LINEAR_ACTION_SCALE * forward_action) * (
                        1 + NOISE_SCALE * jax.random.normal(noise_keys[0], ()))
            ang_speed = WOLF_ANGULAR_SPEED_SCALE * angular_action * (1 + NOISE_SCALE * jax.random.normal(noise_keys[1], ()))

            # updated positions
            #x_new = jnp.clip(x + dt * x_dot, -x_max_arena, x_max_arena)  # no wrap-around for now; may change it later
            #y_new = jnp.clip(y + dt * y_dot, -y_max_arena, y_max_arena)

            x_new = jnp.mod(x + dt * x_dot + x_max_arena, 2 * x_max_arena) - x_max_arena
            y_new = jnp.mod(y + dt * y_dot + y_max_arena, 2 * y_max_arena) - y_max_arena
            ang_new = jnp.mod(ang + dt * ang_dot + jnp.pi, 2 * jnp.pi) - jnp.pi

            x_dot_new = speed * jnp.cos(ang) - dt * x_dot * damping
            y_dot_new = speed * jnp.sin(ang) - dt * y_dot * damping
            ang_dot_new = ang_speed - dt * ang_dot * damping

            # metabolic cost
            metabolic_cost = BASIC_METABOLIC_COST_WOLF
            energy_new = energy + energy_intake - metabolic_cost

            agent_is_dead = energy_new[0] <= 0.0

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new,
                                 "ang_dot": ang_dot_new,
                                 "energy": energy_new}
            new_state = State(content=new_state_content)
            return jax.lax.cond(
                agent_is_dead,
                lambda _: agent.replace(state=new_state, active_state=0),  # mark as dead/inactive
                lambda _: agent.replace(state=new_state, key=key, age=agent.age + dt, policy=new_policy),
                None
            )
        def step_inactive_agent():
            return agent

        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)

    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]
        key = agent.key

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        energy = random.uniform(subkeys[3], shape=(1,), minval=0.5 * energy_begin_max, maxval=energy_begin_max)

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                         "energy": energy}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key)



def calculate_sheep_energy_intake(sheep: Sheep):
    """Divide energy supply that a sheep receives by the number of sheep (including themselves) in their energy radius.
        Returns the energy intake for a sheep.
    """
    def sheep_local_density(one_sheep, all_sheep):
        """Calculate how many other sheep are within this sheep's energy radius"""
        xs_sheep = all_sheep.state.content["x"].reshape(-1)
        ys_sheep = all_sheep.state.content["y"].reshape(-1)
        active_sheep = all_sheep.active_state.astype(bool)

        x_sheep = one_sheep.state.content["x"]
        y_sheep = one_sheep.state.content["y"]

        # calculate distance to all other sheep
        distances = jnp.linalg.norm(jnp.stack((xs_sheep - x_sheep, ys_sheep - y_sheep), axis=1), axis=1).reshape(-1)

        energy_radius = SHEEP_RADIUS* 3.0

        cond = jnp.logical_and(distances <= energy_radius, active_sheep)
        is_near = jnp.where(cond, 1.0, 0.0)
        num_sheep_in_radius = jnp.sum(is_near)

        return num_sheep_in_radius, is_near

    active_mask = sheep.active_state.astype(bool)

    num_sheep_in_radius, is_near_matrix = jax.vmap(sheep_local_density, in_axes=(0, None))(sheep, sheep)

    energy_share = jnp.divide(BASE_ENERGY_RATE, jnp.maximum(num_sheep_in_radius, 1.0))
    energy_intake = energy_share * active_mask

    return energy_intake

jit_calculate_sheep_energy_intake = jax.jit(calculate_sheep_energy_intake)

def wolves_sheep_interactions(sheep: Sheep, wolves: Wolf):
    def wolf_sheep_interaction(one_wolf, sheep):
        xs_sheep = sheep.state.content["x"]
        ys_sheep = sheep.state.content["y"]
        x_wolf = one_wolf.state.content["x"]
        y_wolf = one_wolf.state.content["y"]

        active_sheep = sheep.active_state

        wolf_radius = one_wolf.params.content["radius"]

        distances = jnp.linalg.norm(jnp.stack((xs_sheep - x_wolf, ys_sheep - y_wolf), axis=1), axis=1).reshape(-1)
        is_in_range = jnp.where(jnp.logical_and(distances <= wolf_radius, active_sheep), 1.0, 0.0) # only consider active sheep

        # find the closest sheep; wolf can only catch one sheep at a time
        distances_masked = jnp.where(is_in_range > 0, distances, jnp.inf)
        closest_sheep_idx = jnp.argmin(distances_masked)

        is_catching_sheep = jnp.zeros_like(is_in_range)
        is_catching_sheep = is_catching_sheep.at[closest_sheep_idx].set(
            jnp.where(distances_masked[closest_sheep_idx] < jnp.inf, 1.0, 0.0)
        )
        return is_catching_sheep

    is_catching_matrix = jax.vmap(wolf_sheep_interaction, in_axes=(0, None))(wolves, sheep) # shape (num_wolves, num_sheep)
    is_being_fed_on = jnp.any(is_catching_matrix, axis=0)  # shape (num_sheep,) - t/f if sheep is being fed on by any wolf

    #split energy among wolves if multiple wolves target same sheep
    num_wolves_at_sheep = jnp.maximum(jnp.sum(is_catching_matrix, axis=0), 1.0)
    energy_sharing_matrix = jnp.divide(is_catching_matrix, num_wolves_at_sheep)

    energy_offer_per_sheep = sheep.state.content["energy"].reshape(-1) * EAT_RATE_SHEEP

    # calculate energy intake for each wolf
    energy_intake_wolves = jnp.multiply(energy_sharing_matrix, energy_offer_per_sheep)
    energy_intake_wolves = jnp.sum(energy_intake_wolves, axis=1).reshape(-1)

    # calculate energy loss for each sheep
    energy_loss_sheep = jnp.where(is_being_fed_on, energy_offer_per_sheep, 0.0)

    return energy_loss_sheep, energy_intake_wolves

jit_wolves_sheep_interactions = jax.jit(wolves_sheep_interactions)

@struct.dataclass
class CTRNN(Policy):
    @staticmethod
    def create_policy(params: Params, key: jax.random.PRNGKey):
        num_neurons = params.content["num_neurons"]
        num_obs = params.content["num_obs"]
        num_actions = params.content["num_actions"]

        Z = jnp.zeros((num_neurons,), dtype=jnp.float32)
        action = jnp.zeros((num_actions,), dtype=jnp.float32)
        state = State(content={'Z': Z, 'action': action})

        J = jnp.zeros((num_neurons, num_neurons), dtype=jnp.float32)
        E = jnp.zeros((num_neurons, num_obs), dtype=jnp.float32)
        D = jnp.zeros((num_actions, num_neurons), dtype=jnp.float32)
        tau = jnp.zeros((num_neurons,), dtype=jnp.float32)
        B = jnp.zeros((num_neurons,), dtype=jnp.float32)

        params = Params(content={'J': J, 'E': E, 'D': D, 'tau': tau, 'B': B})
        return Policy(params=params, state=state, key=key)

    @staticmethod
    @jax.jit
    def step_policy(policy: Policy, input: Signal, params: Params):
        dt = params.content["dt"]
        action_scale = params.content["action_scale"]
        time_constant_scale = params.content["time_constant_scale"]

        J = policy.params.content["J"]
        E = policy.params.content["E"]
        D = policy.params.content["D"]
        tau = policy.params.content["tau"]
        B = policy.params.content["B"]

        Z = policy.state.content["Z"]

        obs = input.content["obs"]

        #step the policy
        z_dot = jnp.matmul(J, jnp.tanh(Z + B)) + jnp.matmul(E, obs) - Z
        z_dot = jnp.multiply(z_dot, time_constant_scale * jax.nn.sigmoid(tau))

        new_Z = Z + dt * z_dot # euler integration
        readout = jnp.matmul(D, new_Z)
        actions = action_scale * jnp.array([jax.nn.sigmoid(readout[0]), jax.nn.tanh(readout[1])]) # 0; speed, 1; angular speed

        new_policy_state = State(content={'Z': new_Z, 'action': actions})
        new_policy = policy.replace(state=new_policy_state)
        return new_policy

    @staticmethod
    @jax.jit
    def set_policy(policy: Policy, set_params: Params):
        J = set_params.content['J']
        tau = set_params.content['tau']
        E = set_params.content['E']
        B = set_params.content['B']
        D = set_params.content['D']
        new_policy_params = Params(content={'J': J, 'tau': tau, 'E': E, 'B': B, 'D': D})
        return policy.replace(params=new_policy_params)


# potentially make this into two seperate functions (for sheep and wolves):
def set_CMAES_params(CMAES_params, agents):
    """
    copy the CMAES_params to the agents while manipulating the shape of the parameters
    Args:
        - CMAES_params: The parameters to set with shape (NUM_FORAGERS, NUM_ES_PARAMS)
        - agents: The agents to set the parameters to
    Returns:
        The updated agents
    """
    J = CMAES_params[:,:NUM_NEURONS*NUM_NEURONS].reshape((-1, NUM_NEURONS, NUM_NEURONS))
    last_index = NUM_NEURONS*NUM_NEURONS

    tau = CMAES_params[:, last_index:last_index + NUM_NEURONS].reshape((-1, NUM_NEURONS))
    last_index += NUM_NEURONS

    E = CMAES_params[:, last_index:last_index + NUM_NEURONS * NUM_OBS].reshape((-1, NUM_NEURONS, NUM_OBS))
    last_index += NUM_NEURONS*NUM_OBS

    B = CMAES_params[:, last_index:last_index + NUM_NEURONS].reshape((-1, NUM_NEURONS))
    last_index += NUM_NEURONS

    D = CMAES_params[:, last_index:last_index + NUM_NEURONS * NUM_ACTIONS].reshape((-1, NUM_ACTIONS, NUM_NEURONS))

    policy_params = Params(content={'J': J, 'tau': tau, 'E': E, 'B': B, 'D': D})
    new_policies = jax.vmap(CTRNN.set_policy)(agents.policy, policy_params)
    return agents.replace(policy=new_policies)

jit_set_CMAES_params = jax.jit(set_CMAES_params)




@struct.dataclass
class PredatorPreyWorld:
    sheep_set: Set
    wolf_set: Set

    @staticmethod
    def create_world(params, key):
        sheep_params = params.content["sheep_params"]
        wolf_params = params.content["wolf_params"]
        policy_params_sheep = params.content["policy_params_sheep"]
        policy_params_wolf = params.content["policy_params_wolf"]

        num_sheep = sheep_params["num_sheep"]
        num_wolf = wolf_params["num_wolf"]

        key, *policy_keys = jax.random.split(key, num_sheep + 1)
        policy_keys = jnp.array(policy_keys)

        policy_create_params_sheep = Params(content={'num_neurons': policy_params_sheep['num_neurons'],
                                               'num_obs': policy_params_sheep['num_obs'],
                                               'num_actions': policy_params_sheep['num_actions']})
        policies_sheep = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params_sheep, policy_keys)

        key, *policy_keys = jax.random.split(key, num_wolf + 1)
        policy_keys = jnp.array(policy_keys)
        policy_create_params_wolf = Params(content={'num_neurons': policy_params_wolf['num_neurons'],
                                               'num_obs': policy_params_wolf['num_obs'],
                                               'num_actions': policy_params_wolf['num_actions']})

        policies_wolf = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params_wolf, policy_keys)

        key, sheep_key = random.split(key, 2)

        x_max_array = jnp.tile(jnp.array([sheep_params["x_max"]]), (num_sheep,))
        y_max_array = jnp.tile(jnp.array([sheep_params["y_max"]]), (num_sheep,))
        energy_begin_max_array = jnp.tile(jnp.array([sheep_params["energy_begin_max"]]), (num_sheep,))
        eat_rate_array = jnp.tile(jnp.array([sheep_params["eat_rate"]]), (num_sheep,))
        radius_array = jnp.tile(jnp.array([sheep_params["radius"]]), (num_sheep,))
        mass_array = jnp.tile(jnp.array([sheep_params["mass_begin"]]), (num_sheep,))

        sheep_create_params = Params(content= {
            "x_max": x_max_array,
            "y_max": y_max_array,
            "energy_begin_max": energy_begin_max_array,
            "eat_rate": eat_rate_array,
            "radius": radius_array,
            "mass_begin": mass_array,
            "policy": policies_sheep
        })

        sheep = create_agents(agent=Sheep, params=sheep_create_params, num_agents=num_sheep, num_active_agents=num_sheep,
                              agent_type=sheep_params["agent_type"], key=sheep_key)

        sheep_set = Set(num_agents=num_sheep, num_active_agents=num_sheep, agents=sheep, id=0, set_type=sheep_params["agent_type"],
                        params=None, state=None, policy=None, key=None)

        key, wolf_key = random.split(key, 2)
        x_max_array = jnp.tile(jnp.array([wolf_params["x_max"]]), (num_wolf,))
        y_max_array = jnp.tile(jnp.array([wolf_params["y_max"]]), (num_wolf,))
        energy_begin_max_array = jnp.tile(jnp.array([wolf_params["energy_begin_max"]]), (num_wolf,))
        radius_array = jnp.tile(jnp.array([wolf_params["radius"]]), (num_wolf,))
        mass_array = jnp.tile(jnp.array([wolf_params["mass_begin"]]), (num_wolf,))

        wolf_create_params = Params(content= {"x_max": x_max_array,
                                              "y_max": y_max_array,
                                              "energy_begin_max": energy_begin_max_array,
                                              "radius": radius_array,
                                              "mass_begin": mass_array,
                                              "policy": policies_wolf
        })

        wolves = create_agents(agent=Wolf, params=wolf_create_params, num_agents=num_wolf, num_active_agents=num_wolf,
                               agent_type=wolf_params["agent_type"], key=wolf_key)

        wolf_set = Set(num_agents=num_wolf, num_active_agents=num_wolf, agents=wolves, id=2, set_type=wolf_params["agent_type"],
                       params=None, state=None, policy=None, key=None)


        return PredatorPreyWorld(sheep_set=sheep_set, wolf_set=wolf_set)


def step_world(pred_prey_world, _t):
    sheep_set = pred_prey_world.sheep_set
    wolf_set = pred_prey_world.wolf_set

    sheep_sensor_data, wolf_sensor_data = jit_get_all_agent_sensors(
        sheep_set.agents, wolf_set.agents, SHEEP_AGENT_TYPE, WOLF_AGENT_TYPE)

    energy_intake_from_environment = jit_calculate_sheep_energy_intake(sheep_set.agents)
    energy_loss_sheep, energy_intake_wolves = jit_wolves_sheep_interactions(sheep_set.agents, wolf_set.agents)

    sheep_step_input = Signal(content={'obs': sheep_sensor_data, "energy_intake": energy_intake_from_environment - energy_loss_sheep})
    sheep_step_params = Params(content={"dt": Dt,
                                        "damping": DAMPING,
                                        "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                        "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                        "x_max_arena": WORLD_SIZE_X,
                                        "y_max_arena": WORLD_SIZE_Y,
                                        "action_scale": ACTION_SCALE,
                                        "time_constant_scale": TIME_CONSTANT_SCALE
    })
    sheep_set = jit_step_agents(Sheep.step_agent, sheep_step_params, sheep_step_input, sheep_set)


    wolf_step_input = Signal(content={"obs": wolf_sensor_data, "energy_intake": energy_intake_wolves})
    wolf_step_params = Params(content={"dt": Dt,
                                       "damping": DAMPING,
                                       "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                       "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                       "x_max_arena": WORLD_SIZE_X,
                                       "y_max_arena": WORLD_SIZE_Y,
                                       "action_scale": ACTION_SCALE,
                                       "time_constant_scale": TIME_CONSTANT_SCALE
    })
    wolf_set = jit_step_agents(Wolf.step_agent, wolf_step_params, wolf_step_input, wolf_set)


    render_data = Signal(content={"sheep_xs": sheep_set.agents.state.content["x"].reshape(-1, 1),
                                  "sheep_ys": sheep_set.agents.state.content["y"].reshape(-1, 1),
                                  "sheep_angles": sheep_set.agents.state.content["ang"].reshape(-1, 1),
                                  "sheep_energy": sheep_set.agents.state.content["energy"].reshape(-1, 1),
                                  "wolf_xs": wolf_set.agents.state.content["x"].reshape(-1, 1),
                                  "wolf_ys": wolf_set.agents.state.content["y"].reshape(-1, 1),
                                  "wolf_angles": wolf_set.agents.state.content["ang"].reshape(-1, 1),
                                  "wolf_energy": wolf_set.agents.state.content["energy"].reshape(-1, 1)
    })

    return pred_prey_world.replace(sheep_set=sheep_set, wolf_set=wolf_set), render_data

jit_step_world = jax.jit(step_world)


def reset_world(pred_prey_world):
    sheep_set_agents = pred_prey_world.sheep_set.agents
    wolf_set_agents = pred_prey_world.wolf_set.agents

    sheep_set_agents = jax.vmap(Sheep.reset_agent)(sheep_set_agents, None)
    wolf_set_agents = jax.vmap(Wolf.reset_agent)(wolf_set_agents, None)

    sheep_set = pred_prey_world.sheep_set.replace(agents=sheep_set_agents)
    wolf_set = pred_prey_world.wolf_set.replace(agents=wolf_set_agents)

    return pred_prey_world.replace(sheep_set=sheep_set, wolf_set=wolf_set)

jit_reset_world = jax.jit(reset_world)


def scan_episode(pred_prey_world: PredatorPreyWorld, ts):
    return jax.lax.scan(jit_step_world, pred_prey_world, ts)

jit_scan_episode = jax.jit(scan_episode)

def run_episode(pred_prey_world: PredatorPreyWorld):
    ts = jnp.arange(EP_LEN)
    pred_prey_world = jit_reset_world(pred_prey_world)
    pred_prey_world, render_data = jit_scan_episode(pred_prey_world, ts)
    render_data = Signal(content={
        "sheep_xs": render_data.content["sheep_xs"],
        "sheep_ys": render_data.content["sheep_ys"],
        "sheep_angles": render_data.content["sheep_angles"],
        "sheep_energy": render_data.content["sheep_energy"], ##
        "wolf_xs": render_data.content["wolf_xs"],
        "wolf_ys": render_data.content["wolf_ys"],
        "wolf_angles": render_data.content["wolf_angles"],
        "wolf_energy": render_data.content["wolf_energy"] ##
    })
    return pred_prey_world, render_data

jit_run_episode = jax.jit(run_episode)

def get_energy(CMAES_params_sheep, CMAES_params_wolf, pred_prey_worlds):
    """
    Args:
        - CMAES_params_sheep: shape (NUM_SHEEP, NUM_ES_PARAMS)
        - CMAES_params_wolf: shape (NUM_WOLF, NUM_ES_PARAMS)
        - pred_prey_worlds: array of worlds, shape (NUM_WORLDS,)
    Returns:
        - sheep_energy: shape (NUM_SHEEP,) - fitness per sheep
        - wolf_energy: shape (NUM_WOLF,) - fitness per wolf
        - pred_prey_worlds: updated worlds
    """
    def update_single_world(world):
        new_sheep_agents = jit_set_CMAES_params(CMAES_params_sheep, world.sheep_set.agents)
        new_sheep_set = world.sheep_set.replace(agents=new_sheep_agents)

        new_wolf_agents = jit_set_CMAES_params(CMAES_params_wolf, world.wolf_set.agents)
        new_wolf_set = world.wolf_set.replace(agents=new_wolf_agents)

        return world.replace(sheep_set=new_sheep_set, wolf_set=new_wolf_set)

    pred_prey_worlds = jax.vmap(update_single_world)(pred_prey_worlds)
    pred_prey_worlds, render_data = jax.vmap(jit_run_episode)(pred_prey_worlds)
    # get energy per individual, averaged across worlds
    sheep_energy = jnp.mean(pred_prey_worlds.sheep_set.agents.state.content["energy"], axis=0).reshape(-1) # mean axis=0 -> (NUM_SHEEP,)
    wolf_energy = jnp.mean(pred_prey_worlds.wolf_set.agents.state.content["energy"], axis=0).reshape(-1)

    return sheep_energy, wolf_energy, pred_prey_worlds, render_data # use this

jit_get_energy = jax.jit(get_energy)




def main():
    key, *pred_prey_world_keys = random.split(KEY, NUM_WORLDS+1)
    pred_prey_world_keys = jnp.array(pred_prey_world_keys)

    pred_prey_worlds = jax.vmap(PredatorPreyWorld.create_world, in_axes=(None,0))(PP_WORLD_PARAMS, pred_prey_world_keys)

    key, sheep_key, wolf_key = random.split(key, 3)
    dummy_solution = jnp.zeros(NUM_ES_PARAMS) # is this right?

    strategy_sheep = CMA_ES(population_size=NUM_SHEEP, solution=dummy_solution)
    es_params_sheep = strategy_sheep.default_params
    state_sheep = strategy_sheep.init(key=sheep_key, mean=dummy_solution, params=es_params_sheep)

    strategy_wolf = CMA_ES(population_size=NUM_WOLF, solution=dummy_solution)
    es_params_wolf = strategy_wolf.default_params
    state_wolf = strategy_wolf.init(key=wolf_key, mean= dummy_solution, params=es_params_wolf)

    mean_sheep_energy_list = []
    mean_wolf_energy_list = []
    sheep_param_list = []
    wolf_param_list = []

    # for rendering/plotting:
    sheep_xs_list, sheep_ys_list, sheep_angles_list, sheep_energy_list = [], [], [], []
    wolf_xs_list, wolf_ys_list, wolf_angles_list, wolf_energy_list = [], [], [], []
    energy_thresh_save = ENERGY_THRESH_SAVE

    print(f"Starting co-evolution with {NUM_WORLDS} worlds, {NUM_GENERATIONS} generations")
    print(f"Sheep population: {NUM_SHEEP}, Wolf population: {NUM_WOLF}")

    for generation in range(NUM_GENERATIONS):
        key, sheep_gen_key, wolf_gen_key = jax.random.split(key, 3)

        # get candidate solutions: (NUM_SHEEP, NUM_ES_PARAMS) and (NUM_WOLF, NUM_ES_PARAMS)
        x_sheep, state_sheep = strategy_sheep.ask(sheep_gen_key, state_sheep, es_params_sheep)
        x_wolf, state_wolf = strategy_wolf.ask(wolf_gen_key, state_wolf, es_params_wolf)

        # run simulation once and get both energy and render data
        sheep_energy, wolf_energy, pred_prey_worlds, render_data_all = jit_get_energy(x_sheep, x_wolf, pred_prey_worlds)

        state_sheep = strategy_sheep.tell(sheep_gen_key, x_sheep, -1*sheep_energy, state_sheep, es_params_sheep) # am I using the right keys?
        state_wolf = strategy_wolf.tell(wolf_gen_key, x_wolf, -1*wolf_energy, state_wolf, es_params_wolf)

        mean_sheep_energy = jnp.mean(sheep_energy)
        best_sheep_energy = jnp.max(sheep_energy)
        worst_sheep_energy = jnp.min(sheep_energy)

        mean_wolf_energy = jnp.mean(wolf_energy)
        best_wolf_energy = jnp.max(wolf_energy)
        worst_wolf_energy = jnp.min(wolf_energy)

        mean_sheep_energy_list.append(mean_sheep_energy)
        mean_wolf_energy_list.append(mean_wolf_energy)

        # save parameters
        if mean_sheep_energy > energy_thresh_save:
            energy_thresh_save += ENERGY_THRESH_SAVE_STEP
            sheep_param_list.append(state_sheep.mean)
            wolf_param_list.append(state_wolf.mean)
            print(f"  Saved parameters at fitness {mean_sheep_energy:.2f}")

        #_, render_data_all = jax.vmap(jit_run_episode)(pred_prey_worlds)

        # extract sheep/grass data from all worlds at once
        sheep_xs_list.append(render_data_all.content["sheep_xs"])
        sheep_ys_list.append(render_data_all.content["sheep_ys"])
        sheep_angles_list.append(render_data_all.content["sheep_angles"])
        sheep_energy_list.append(render_data_all.content["sheep_energy"])

        wolf_xs_list.append(render_data_all.content["wolf_xs"])
        wolf_ys_list.append(render_data_all.content["wolf_ys"])
        wolf_angles_list.append(render_data_all.content["wolf_angles"])
        wolf_energy_list.append(render_data_all.content["wolf_energy"])

        print(f'Generation: {generation}, '
              f'Sheep - Mean: {mean_sheep_energy:.2f}, Best: {best_sheep_energy:.2f}, Worst: {worst_sheep_energy:.2f}, '
              f'Wolf - Mean: {mean_wolf_energy:.2f}, Best: {best_wolf_energy:.2f}, Worst: {worst_wolf_energy:.2f}')


    # convert to arrays and save
    os.makedirs(DATA_PATH, exist_ok=True)

    jnp.save(DATA_PATH + 'mean_sheep_energy_list.npy',jnp.array(mean_sheep_energy_list))
    jnp.save(DATA_PATH + 'mean_wolf_energy_list.npy', jnp.array(mean_wolf_energy_list))
    jnp.save(DATA_PATH + 'final_key.npy', jnp.array(key))

    # save sheep rendering-plotting data
    jnp.save(DATA_PATH + 'rendering_sheep_xs.npy', jnp.array(sheep_xs_list))
    jnp.save(DATA_PATH + 'rendering_sheep_ys.npy', jnp.array(sheep_ys_list))
    jnp.save(DATA_PATH + 'rendering_sheep_angs.npy', jnp.array(sheep_angles_list))
    jnp.save(DATA_PATH + 'rendering_sheep_energy.npy',  jnp.array(sheep_energy_list)) # plot energy against ts

    # save wolf rendering-plotting data
    jnp.save(DATA_PATH + 'rendering_wolf_xs.npy', jnp.array(wolf_xs_list))
    jnp.save(DATA_PATH + 'rendering_wolf_ys.npy', jnp.array(wolf_ys_list))
    jnp.save(DATA_PATH + 'rendering_wolf_angs.npy', jnp.array(wolf_angles_list))
    jnp.save(DATA_PATH + 'rendering_wolf_energy.npy', jnp.array(wolf_energy_list)) # plot energy against ts

    jnp.save(DATA_PATH + 'sheep_param_list.npy', jnp.array(sheep_param_list))
    jnp.save(DATA_PATH + 'wolf_param_list.npy', jnp.array(wolf_param_list))

    print(f"Simulation completed. Data saved to {DATA_PATH}")


if __name__ == "__main__":
    main()



















