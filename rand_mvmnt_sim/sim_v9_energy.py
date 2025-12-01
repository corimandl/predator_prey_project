"""
9th version of AB predator-prey simulation

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

- Removed grass patches
- Instead; use K-NN distance = average distance to K nearest neighbors to detect crowding/spreading of sheep
-          Constant energy supply; Energy circle; inside circle -> agents get energy, outside -> no energy supply

"""
import os
from structs import *
from functions import *

import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
#from evosax import CMA_ES

WORLD_SIZE_X = 10000.0
WORLD_SIZE_Y = 10000.0

MAX_SPAWN_X = 500.0
MAX_SPAWN_Y = 500.0

Dt = 0.1 # discrete time increments
KEY = jax.random.PRNGKey(0) # which seed?
NOISE_SCALE = 0.05
DAMPING = 0.1

# Sheep parameters
NUM_SHEEP = 100
SHEEP_RADIUS = 5.0
SHEEP_ENERGY_BEGIN_MAX = 50.0
SHEEP_MASS_BEGIN = 5.0 # initial sheep mass at birth
SHEEP_AGENT_TYPE = 1
EAT_RATE_SHEEP = 0.4

# Wolf parameters
NUM_WOLF = 100
WOLF_RADIUS = 7.0
WOLF_ENERGY_BEGIN_MAX = 50.0
WOLF_MASS_BEGIN = 7.0
WOLF_AGENT_TYPE = 3

# Metabolism
METABOLIC_COST_SPEED = 0.01
METABOLIC_COST_ANGULAR = 0.05
BASIC_METABOLIC_COST_SHEEP = 0.02
BASIC_METABOLIC_COST_WOLF = 0.04

# Energy circle parameters (replaces grass)
ENERGY_CIRCLE_RADIUS = 300.0  # energy available within this radius from center
BASE_ENERGY_RATE = 0.5  # base energy gain per timestep
K_NEIGHBORS = 5  # number of nearest neighbors to check
OPTIMAL_KNN_DISTANCE = 50.0  # ideal spacing between sheep
KNN_PENALTY_SCALE = 0.01  # how much crowding reduces energy

# Action parameters (sheep)
ACTION_SCALE = 1.0
LINEAR_ACTION_OFFSET = 0.0

SHEEP_SPEED_MULTIPLIER = 2.0
SHEEP_LINEAR_ACTION_SCALE = SHEEP_SPEED_MULTIPLIER * SHEEP_RADIUS / Dt
SHEEP_ANGULAR_SPEED_SCALE = 5.0

# Action parameters (wolves)
WOLF_SPEED_MULTIPLIER = 2.5
WOLF_LINEAR_ACTION_SCALE = WOLF_SPEED_MULTIPLIER * WOLF_RADIUS / Dt
WOLF_ANGULAR_SPEED_SCALE = 5.0


# Training parameters
NUM_WORLDS = 10
NUM_GENERATIONS = 10
EP_LEN = 500

FITNESS_THRESH_SAVE = 150.0 # threshold for saving render data
FITNESS_THRESH_SAVE_STEP = 10.0 # the amount by which we increase the threshold for saving render data

# save data
DATA_PATH = "./data/sheep_wolf_data1/"


# Predator-prey world parameters
PP_WORLD_PARAMS = Params(content= {"sheep_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_begin_max": SHEEP_ENERGY_BEGIN_MAX,
                                                    "mass_begin": SHEEP_MASS_BEGIN,
                                                    "eat_rate": EAT_RATE_SHEEP,
                                                    "radius": SHEEP_RADIUS,
                                                    "agent_type": SHEEP_AGENT_TYPE,
                                                    "num_sheep": NUM_SHEEP,
                                                    },
                                   "wolf_params": {"x_max": MAX_SPAWN_X,
                                                   "y_max": MAX_SPAWN_Y,
                                                   "energy_begin_max": WOLF_ENERGY_BEGIN_MAX,
                                                   "mass_begin": WOLF_MASS_BEGIN,
                                                   "radius": WOLF_RADIUS,
                                                   "agent_type": WOLF_AGENT_TYPE,
                                                   "num_wolf": NUM_WOLF,
                                                   }
                                   })


# Sheep dataclass
@struct.dataclass
class Sheep(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
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
            fitness = jnp.array([0.0])
            energy_offer = energy * eat_rate

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy, "fitness": fitness, "energy_offer": energy_offer}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([-1.0]), "y": jnp.array([-1.0]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([-1.0]), "fitness": jnp.array([-1.0]), "energy_offer": jnp.array([-1.0])} # placeholder values
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)

        return Sheep(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=None)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            # input
            energy_intake = input.content["energy_intake"] # also handles energy output (if eaten by wolves)

            # current agent state
            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]

            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"] # current x_velocity
            y_dot = agent.state.content["y_dot"] # current y_velocity
            ang_dot = agent.state.content["ang_dot"]

            dt = step_params.content["dt"]
            damping = step_params.content["damping"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            eat_rate = agent.params.content["eat_rate"]

            key, *subkeys = random.split(agent.key, 5)

            # sample random movement
            forward_action = jax.random.uniform(subkeys[0], (), minval=0.0, maxval=1.0)
            angular_action = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=1.0)

            # fixed base speed (with noise)
            speed = ((LINEAR_ACTION_OFFSET + SHEEP_LINEAR_ACTION_SCALE * forward_action) *
                     (1 + NOISE_SCALE * jax.random.normal(subkeys[2], ())))
            ang_speed = SHEEP_ANGULAR_SPEED_SCALE * angular_action * (1 + NOISE_SCALE * jax.random.normal(subkeys[3], ()))

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

            new_energy_offer = energy_new * eat_rate
            fitness_new = energy_new # for now: fitness is energy

            agent_is_dead = energy_new[0] <= 0.0

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new, "ang_dot": ang_dot_new,
                                 "energy": energy_new, "fitness": fitness_new, "energy_offer": new_energy_offer}
            new_state = State(content=new_state_content)

            return jax.lax.cond(
                agent_is_dead,
                lambda _: agent.replace(state=new_state, active_state=0),  # mark as dead/inactive
                lambda _: agent.replace(state=new_state, key=key, age=agent.age + dt),
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
        energy_offer = eat_rate * energy
        fitness = jnp.array([0.0])

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                         "energy": energy, "fitness": fitness, "energy_offer": energy_offer}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key)


@struct.dataclass
class Wolf(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_begin_max = params.content["energy_begin_max"]
        radius = params.content["radius"]
        mass_begin = params.content["mass_begin"]

        key, *subkeys = random.split(key, 5)
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
            fitness = jnp.array([0.0])

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                             "energy": energy, "fitness": fitness}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([-1.0]), "y": jnp.array([-1.0]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([-1.0]), "fitness": jnp.array([-1.0])}
            state = State(content=state_content)
            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)
        return Wolf(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=None)

    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            energy_intake = input.content["energy_intake"]

            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"]  # current x_velocity
            y_dot = agent.state.content["y_dot"]  # current y_velocity
            ang_dot = agent.state.content["ang_dot"]

            dt = step_params.content["dt"]
            damping = step_params.content["damping"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            key, *subkeys = random.split(agent.key, 5)

            # sample random movement
            forward_action = jax.random.uniform(subkeys[0], (), minval=0.0, maxval=1.0)
            angular_action = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=1.0)

            # fixed base speed (with noise)
            speed = (LINEAR_ACTION_OFFSET + WOLF_LINEAR_ACTION_SCALE * forward_action) * (
                        1 + NOISE_SCALE * jax.random.normal(subkeys[2], ()))
            ang_speed = WOLF_ANGULAR_SPEED_SCALE * angular_action * (1 + NOISE_SCALE * jax.random.normal(subkeys[3], ()))

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
            fitness_new = energy_new  # for now: fitness is energy

            agent_is_dead = energy_new[0] <= 0.0

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new,
                                 "ang_dot": ang_dot_new,
                                 "energy": energy_new, "fitness": fitness_new}
            new_state = State(content=new_state_content)
            return jax.lax.cond(
                agent_is_dead,
                lambda _: agent.replace(state=new_state, active_state=0),  # mark as dead/inactive
                lambda _: agent.replace(state=new_state, key=key, age=agent.age + dt),
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
        fitness = jnp.array([0.0])

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot,
                         "energy": energy, "fitness": fitness}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key)

# K-NN based energy system (replaces grass)
def calculate_sheep_energy_intake(sheep: Sheep):
    """
    Energy based on:
    1. Being inside the energy circle
    2. K-NN distance (spacing from neighbors)
    """
    # extract positions - only consider active sheep
    active_mask = sheep.active_state.astype(bool)
    x_positions = sheep.state.content["x"].reshape(-1)
    y_positions = sheep.state.content["y"].reshape(-1)

    num_sheep = x_positions.shape[0]

    # 1. check if inside energy circle
    distances_from_center = jnp.sqrt(x_positions ** 2 + y_positions ** 2)
    in_energy_zone = (distances_from_center < ENERGY_CIRCLE_RADIUS).astype(jnp.float32)

    # 2. calculate pairwise distances between all sheep
    positions = jnp.stack([x_positions, y_positions], axis=1) # shape: (num_sheep, 2)
    diff = positions[:, None, :] - positions[None, :, :]  # broadcast to (num_sheep, num_sheep, 2)
    pairwise_distances = jnp.linalg.norm(diff, axis=2)  # shape: (num_sheep, num_sheep)

    # 3. for each sheep, find K nearest neighbors
    self_mask = jnp.eye(num_sheep, dtype=bool) # exclude comparing sheep with itself (identity matrix with diagonal True)
    inactive_mask = ~active_mask[None, :] | ~active_mask[:, None]  # compare all sheep pairs and mark any pair where at least one sheep is inactive
    exclude_mask = self_mask | inactive_mask # pairs that are excluded

    pairwise_distances = jnp.where(exclude_mask, jnp.inf, pairwise_distances) # set self-distance and inactive sheep to inf to exclude them (if exclude_mask = True)

    # get K nearest neighbors
    knn_distances = jnp.sort(pairwise_distances, axis=1)[:, :K_NEIGHBORS] # how many K_NEIGHBORS do we pick?

    # 4. calculate mean K-NN distance for each sheep
    valid_distances = jnp.where(jnp.isfinite(knn_distances), knn_distances, 0.0) # keep distance if finite, else set 0
    num_valid_neighbors = jnp.sum(jnp.isfinite(knn_distances), axis=1) # count True value per sheep
    mean_knn_distance = jnp.sum(valid_distances, axis=1) / jnp.maximum(num_valid_neighbors, 1.0) # handle case where there are fewer than K neighbors (use mean of available neighbors)

    # 5. energy penalty based on how far from optimal spacing
    # closer than optimal => crowded => penalty
    spacing_penalty = jnp.maximum(0.0, OPTIMAL_KNN_DISTANCE - mean_knn_distance)
    energy_multiplier = jnp.maximum(0.1, 1.0 - KNN_PENALTY_SCALE * spacing_penalty) # 0.1 so sheep don't starve instantly from crowding

    # 6. final energy = base rate * spacing multiplier * zone check * active check
    energy_intake = BASE_ENERGY_RATE * energy_multiplier * in_energy_zone * active_mask.astype(jnp.float32)

    return energy_intake

jit_calculate_sheep_energy_intake = jax.jit(calculate_sheep_energy_intake)


def wolves_sheep_interactions(sheep: Sheep, wolves: Wolf):
    def wolf_sheep_interaction(one_wolf, sheep):
        xs_sheep = sheep.state.content["x"]
        ys_sheep = sheep.state.content["y"]
        x_wolf = one_wolf.state.content["x"]
        y_wolf = one_wolf.state.content["y"]

        wolf_radius = one_wolf.params.content["radius"]
        sheep_radius = sheep.params.content["radius"]

        distances = jnp.linalg.norm(jnp.stack((xs_sheep - x_wolf, ys_sheep - y_wolf), axis=1), axis=1).reshape(-1)
        is_in_range = jnp.where(distances < wolf_radius, 1.0, 0.0)

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
# -------



@struct.dataclass
class PredatorPreyWorld:
    sheep_set: Set
    wolf_set: Set

    @staticmethod
    def create_world(params, key):
        sheep_params = params.content["sheep_params"]
        wolf_params = params.content["wolf_params"]

        num_sheep = sheep_params["num_sheep"]

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
            "mass_begin": mass_array
        })

        sheep = create_agents(agent=Sheep, params=sheep_create_params, num_agents=num_sheep, num_active_agents=num_sheep,
                              agent_type=sheep_params["agent_type"], key=sheep_key)

        sheep_set = Set(num_agents=num_sheep, num_active_agents=num_sheep, agents=sheep, id=0, set_type=sheep_params["agent_type"],
                        params=None, state=None, policy=None, key=None)


        num_wolf = wolf_params["num_wolf"]
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
                                              "mass_begin": mass_array
        })

        wolves = create_agents(agent=Wolf, params=wolf_create_params, num_agents=num_wolf, num_active_agents=num_wolf,
                               agent_type=wolf_params["agent_type"], key=wolf_key)

        wolf_set = Set(num_agents=num_wolf, num_active_agents=num_wolf, agents=wolves, id=2, set_type=wolf_params["agent_type"],
                       params=None, state=None, policy=None, key=None)


        return PredatorPreyWorld(sheep_set=sheep_set, wolf_set=wolf_set)


def step_world(pp_world, _t):
    sheep_set = pp_world.sheep_set
    wolf_set = pp_world.wolf_set

    energy_intake_from_environment = jit_calculate_sheep_energy_intake(sheep_set.agents)
    energy_loss_sheep, energy_intake_wolves = jit_wolves_sheep_interactions(sheep_set.agents, wolf_set.agents)


    sheep_step_input = Signal(content={"energy_intake": energy_intake_from_environment - energy_loss_sheep})
    sheep_step_params = Params(content={"dt": Dt,
                                        "damping": DAMPING,
                                        "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                        "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                        "x_max_arena": WORLD_SIZE_X,
                                        "y_max_arena": WORLD_SIZE_Y,
    })
    sheep_set = jit_step_agents(Sheep.step_agent, sheep_step_params, sheep_step_input, sheep_set)


    wolf_step_input = Signal(content={"energy_intake": energy_intake_wolves})
    wolf_step_params = Params(content={"dt": Dt,
                                       "damping": DAMPING,
                                       "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                       "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                       "x_max_arena": WORLD_SIZE_X,
                                       "y_max_arena": WORLD_SIZE_Y,
    })
    wolf_set = jit_step_agents(Wolf.step_agent, wolf_step_params, wolf_step_input, wolf_set)


    render_data = Signal(content={"sheep_xs": sheep_set.agents.state.content["x"].reshape(-1, 1),
                                  "sheep_ys": sheep_set.agents.state.content["y"].reshape(-1, 1),
                                  "sheep_angles": sheep_set.agents.state.content["ang"].reshape(-1, 1),
                                  "wolf_xs": wolf_set.agents.state.content["x"].reshape(-1, 1),
                                  "wolf_ys": wolf_set.agents.state.content["y"].reshape(-1, 1),
                                  "wolf_angles": wolf_set.agents.state.content["ang"].reshape(-1, 1)
    })

    return pp_world.replace(sheep_set=sheep_set, wolf_set=wolf_set), render_data

jit_step_world = jax.jit(step_world)


def reset_world(pp_world):
    sheep_set_agents = pp_world.sheep_set.agents
    wolf_set_agents = pp_world.wolf_set.agents

    sheep_set_agents = jax.vmap(Sheep.reset_agent)(sheep_set_agents, None)
    wolf_set_agents = jax.vmap(Wolf.reset_agent)(wolf_set_agents, None)

    sheep_set = pp_world.sheep_set.replace(agents=sheep_set_agents)
    wolf_set = pp_world.wolf_set.replace(agents=wolf_set_agents)

    return pp_world.replace(sheep_set=sheep_set, wolf_set=wolf_set)

jit_reset_world = jax.jit(reset_world)


def scan_episode(pp_world: PredatorPreyWorld, ts):
    return jax.lax.scan(jit_step_world, pp_world, ts)

jit_scan_episode = jax.jit(scan_episode)

def run_episode(pp_world: PredatorPreyWorld):
    ts = jnp.arange(EP_LEN)
    pp_world = jit_reset_world(pp_world)
    pp_world, render_data = jit_scan_episode(pp_world, ts)
    render_data = Signal(content={
        "sheep_xs": render_data.content["sheep_xs"],
        "sheep_ys": render_data.content["sheep_ys"],
        "sheep_angles": render_data.content["sheep_angles"],
        "wolf_xs": render_data.content["wolf_xs"],
        "wolf_ys": render_data.content["wolf_ys"],
        "wolf_angles": render_data.content["wolf_angles"]
    })
    return pp_world, render_data

jit_run_episode = jax.jit(run_episode)

def get_fitness(pp_worlds): # also get wolf fitness
    """
    Run episodes and calculate fitness for predator-prey worlds
    Args:
        - pp_worlds: Array of PredatorPreyWorld instances
    Returns:
        - fitness: Mean fitness across all sheep in all worlds
        - pp_worlds: Updated worlds after running episodes
    """
    pp_worlds, render_data = jax.vmap(jit_run_episode)(pp_worlds)
    sheep_fitness = jnp.mean(pp_worlds.sheep_set.agents.state.content["fitness"], axis=0)
    sheep_fitness = jnp.reshape(sheep_fitness, (-1))

    wolf_fitness = jnp.mean(pp_worlds.wolf_set.agents.state.content["fitness"], axis=0)
    wolf_fitness = jnp.reshape(wolf_fitness, (-1))

    return sheep_fitness, wolf_fitness, pp_worlds

jit_get_fitness = jax.jit(get_fitness)


def main():
    key, *pp_world_keys = random.split(KEY, NUM_WORLDS+1)
    pp_world_keys = jnp.array(pp_world_keys)

    pp_worlds = jax.vmap(PredatorPreyWorld.create_world, in_axes=(None,0))(PP_WORLD_PARAMS, pp_world_keys)

    key, subkey = random.split(key)

    mean_sheep_fitness_list = []
    saved_sheep_fitness_list = []

    mean_wolf_fitness_list = []
    saved_wolf_fitness_list = []

    # for plotting:
    sheep_xs_list = []
    sheep_ys_list = []
    sheep_angles_list = []
    wolf_xs_list = []
    wolf_ys_list = []
    wolf_angles_list = []

    fitness_thresh_save = FITNESS_THRESH_SAVE

    print(f"Starting simulation with {NUM_WORLDS} worlds, {NUM_GENERATIONS} generations")

    for generation in range(NUM_GENERATIONS):
        sheep_fitness, wolf_fitness, pp_worlds = jit_get_fitness(pp_worlds) # get wolf_fitness

        mean_sheep_fitness = jnp.mean(sheep_fitness)
        best_sheep_fitness = jnp.max(sheep_fitness)
        worst_sheep_fitness = jnp.min(sheep_fitness)
        mean_sheep_fitness_list.append(mean_sheep_fitness)

        mean_wolf_fitness = jnp.mean(wolf_fitness)
        best_wolf_fitness = jnp.max(wolf_fitness)
        worst_wolf_fitness = jnp.min(wolf_fitness)
        mean_wolf_fitness_list.append(mean_wolf_fitness)

        if mean_sheep_fitness > fitness_thresh_save:
            fitness_thresh_save += FITNESS_THRESH_SAVE_STEP
            saved_sheep_fitness_list.append(mean_sheep_fitness)
            saved_wolf_fitness_list.append(mean_wolf_fitness)

        _, render_data_all = jax.vmap(jit_run_episode)(pp_worlds)

        # extract sheep/grass data from all worlds at once # add wolf data!
        sheep_xs_list.append(render_data_all.content["sheep_xs"])
        sheep_ys_list.append(render_data_all.content["sheep_ys"])
        sheep_angles_list.append(render_data_all.content["sheep_angles"])

        wolf_xs_list.append(render_data_all.content["wolf_xs"])
        wolf_ys_list.append(render_data_all.content["wolf_ys"])
        wolf_angles_list.append(render_data_all.content["wolf_angles"])

        print(f'Generation: {generation}, '
              f'Sheep - Mean: {mean_sheep_fitness:.2f}, Best: {best_sheep_fitness:.2f}, Worst: {worst_sheep_fitness:.2f}, '
              f'Wolf - Mean: {mean_wolf_fitness:.2f}, Best: {best_wolf_fitness:.2f}, Worst: {worst_wolf_fitness:.2f}')


    # save data
    mean_sheep_fitness_array = jnp.array(mean_sheep_fitness_list)
    mean_wolf_fitness_array = jnp.array(mean_wolf_fitness_list)
    saved_sheep_fitness_array = jnp.array(saved_sheep_fitness_list)
    saved_wolf_fitness_array = jnp.array(saved_wolf_fitness_list)

    sheep_xs_array = jnp.array(sheep_xs_list)
    sheep_ys_array = jnp.array(sheep_ys_list)
    sheep_angles_array = jnp.array(sheep_angles_list)
    wolf_xs_array = jnp.array(wolf_xs_list)
    wolf_ys_array = jnp.array(wolf_ys_list)
    wolf_angles_array = jnp.array(wolf_angles_list)

    os.makedirs(DATA_PATH, exist_ok=True)

    jnp.save(DATA_PATH + 'mean_sheep_fitness_list.npy', mean_sheep_fitness_array)
    jnp.save(DATA_PATH + 'mean_wolf_fitness_list.npy', mean_wolf_fitness_array)
    jnp.save(DATA_PATH + 'saved_sheep_fitness_list.npy', saved_sheep_fitness_array)
    jnp.save(DATA_PATH + 'saved_wolf_fitness_list.npy', saved_wolf_fitness_array)
    jnp.save(DATA_PATH + 'final_key.npy', jnp.array(key))

    # save sheep rendering data
    jnp.save(DATA_PATH + 'rendering_sheep_xs.npy', sheep_xs_array)
    jnp.save(DATA_PATH + 'rendering_sheep_ys.npy', sheep_ys_array)
    jnp.save(DATA_PATH + 'rendering_sheep_angs.npy', sheep_angles_array)

    # save wolf rendering data
    jnp.save(DATA_PATH + 'rendering_wolf_xs.npy', wolf_xs_array)
    jnp.save(DATA_PATH + 'rendering_wolf_ys.npy', wolf_ys_array)
    jnp.save(DATA_PATH + 'rendering_wolf_angs.npy', wolf_angles_array)

    print(f"Simulation completed. Data saved to {DATA_PATH}")



if __name__ == "__main__":
    main()



















