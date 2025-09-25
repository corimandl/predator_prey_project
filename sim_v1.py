"""
First version of AB predator-prey simulation

Characteristics:
- One species only (prey; sheep)
- One resource; grass patches
- Agents move randomly (both linear and angular)
- Energy/Mass/Speed trade-offs
"""
from torch import jit

from structs import *
from functions import *

import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct

#from evosax import CMA_ES

MAX_WORLD_X = 10000.0
MAX_WORLD_Y = 10000.0

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

DEATH_THRESHOLD = 5.0 # energy
MIN_DEATH_TIME = 5.0 # time energy needs to be below threshold for the agent to die
REPRODUCTION_THRESHOLD = 30.0 # energy
MIN_REPRODUCTION_TIME = 5.0 # time energy needs to be above threshold for the agent to reproduce
REPRODUCTION_PROB = 0.1

METABOLIC_COST_SPEED = 0.01
METABOLIC_COST_ANGULAR = 0.05
BASIC_METABOLIC_COST = 0.02

# Grass parameters
NUM_GRASS = 100
GRASS_RADIUS = 5.0
GRASS_AGENT_TYPE = 2
ENERGY_STORED_MAX = 5.0
GROWTH_RATE = 0.1
EAT_RATE = 0.3

# Action parameters (sheep)
#NUM_ACTIONS = 2
ACTION_SCALE = 1.0
LINEAR_ACTION_SCALE = ACTION_SCALE * SHEEP_RADIUS / SHEEP_MASS_BEGIN # heavier agents are slower
# but large but light agents are faster
LINEAR_ACTION_OFFSET = 0.0

# Training parameters
NUM_WORLDS = 1
NUM_GENERATIONS = 1
POPULATION_SIZE = NUM_SHEEP
EP_LEN = 500

# Predator-prey world parameters
PP_WORLD_PARAMS = Params(content= {"sheep_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_begin_max": SHEEP_ENERGY_BEGIN_MAX,
                                                    "mass_begin": SHEEP_MASS_BEGIN,
                                                    "radius": SHEEP_RADIUS,
                                                    "agent_type": SHEEP_AGENT_TYPE,
                                                    "num_sheep": NUM_SHEEP,
                                                    "reproduction_prob": REPRODUCTION_PROB,
                                                    "death_threshold": DEATH_THRESHOLD,
                                                    "reproduction_threshold": REPRODUCTION_THRESHOLD,
                                                    "min_death_time": MIN_DEATH_TIME,
                                                    "min_reproduction_time": MIN_REPRODUCTION_TIME
                                                    },
                                   "grass_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_stored_max": ENERGY_STORED_MAX,
                                                    "growth_rate": GROWTH_RATE,
                                                    "eat_rate": EAT_RATE,
                                                    "radius": GRASS_RADIUS,
                                                    "agent_type": GRASS_AGENT_TYPE,
                                                    "num_grass": NUM_GRASS
                                                    }
                                   })


# Grass dataclass + methods
@struct.dataclass
class Grass(Agent):

    @staticmethod
    def create_agent(type, params, id, active_state, key):
        key, *subkeys = random.split(key, 4)

        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_stored_max = params.content["energy_stored_max"]
        eat_rate = params.content["eat_rate"]

        growth_rate = params.content["growth_rate"]
        radius = params.content["radius"]

        # initialise random position
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        # initialise random energy
        energy = random.uniform(subkeys[2], shape=(1,), minval=10e-4, maxval=energy_stored_max)
        energy_offer = energy * eat_rate

        state_content = {"x": x, "y": y, "energy": energy, "energy_offer": energy_offer}
        state = State(content=state_content)

        params_content = {"growth_rate": growth_rate, "eat_rate": eat_rate, "radius": radius, "x_max": x_max, "y_max": y_max, "energy_stored_max": energy_stored_max}
        params = Params(content=params_content)

        return Grass(id=id, state=state, params=params, active_state=active_state, agent_type=type, age = 0.0, key=key)

    @staticmethod
    def step_agent(agent, input, step_params): # where from step_params?
        eat_rate = agent.params.content["eat_rate"]
        growth_rate = agent.params.content["growth_rate"]
        energy_stored_max = agent.params.content["energy_stored_max"]

        dt = step_params.content["dt"]
        is_energy_out = input.content["is_energy_out"] # T/F (whether sheep ate grass or not)

        energy_offer = agent.state.content["energy_offer"]
        energy = agent.state.content["energy"]

        # compute new energy
        new_energy = energy + energy*growth_rate # energy after growth
        new_energy = new_energy - is_energy_out * energy_offer # energy after sheep ate
        new_energy = jnp.clip(new_energy, 10e-4, energy_stored_max) # make sure to stay within bounds (and not <0)

        # update energy offer
        new_energy_offer = new_energy * eat_rate

        new_state_content = {"x": agent.state.content["x"], "y": agent.state.content["y"], "energy": new_energy, "energy_offer": new_energy_offer}
        new_state = State(content=new_state_content)

        return agent.replace(state=new_state, age= agent.age + dt) # everything else remains the same

    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_stored_max = agent.params.content["energy_stored_max"]
        eat_rate = agent.params.content["eat_rate"]
        key = agent.key

        key, *subkeys = random.split(key, 4)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        energy = random.uniform(subkeys[2], shape=(1,), minval=10e-4, maxval=energy_stored_max)
        energy_offer = eat_rate * energy

        state_content = {"x": x, "y": y, "energy": energy, "energy_offer": energy_offer}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, key=key)

# Sheep dataclass
@struct.dataclass
class Sheep(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        x_max = params.content["x_max"]
        y_max = params.content["y_max"]
        energy_begin_max = params.content["energy_begin_max"]
        radius = params.content["radius"]
        mass_begin = params.content["mass_begin"]  # initial mass
        reproduction_prob = params.content["reproduction_prob"]

        key, *subkeys = random.split(key, 5)

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max, "energy_begin_max": energy_begin_max, "mass": mass_begin, "reproduction_prob": reproduction_prob}
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

            state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy, "fitness": fitness, "reproduce": 0, "reproduction_timer": 0.0, "death_timer": 0.0}
            state = State(content=state_content)
            return state

        def create_inactive_agent():
            state_content = {"x": jnp.array([-1.0]), "y": jnp.array([-1.0]), "ang": jnp.array([0.0]),
                             "x_dot": jnp.zeros((1,)), "y_dot": jnp.zeros((1,)), "ang_dot": jnp.zeros((1,)),
                             "energy": jnp.array([-1.0]), "fitness": jnp.array([-1.0]), "reproduce": 0, "reproduction_timer": 0.0, "death_timer": 0.0} # placeholder values
            state = State(content=state_content)

            return state

        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(),
                                   lambda _: create_inactive_agent(), None)

        return Sheep(id=id, state=agent_state, params=params, active_state=active_state, agent_type=type, age=0.0, key=key,policy=None)



    @staticmethod
    def step_agent(agent, input, step_params):

        def step_active_agent():
            # input
            energy_intake = input.content["energy_intake"] # 'energy_in_foragers' in step_world
            is_on_grass = input.content["is_on_grass"] # => compute in 'sheep_grass_interaction' in agent_interactions

            # current agent state
            energy = agent.state.content["energy"]
            fitness = agent.state.content["fitness"]
            x = agent.state.content["x"]
            y = agent.state.content["y"]
            ang = agent.state.content["ang"]
            x_dot = agent.state.content["x_dot"]
            y_dot = agent.state.content["y_dot"]
            ang_dot = agent.state.content["ang_dot"]

            dt = step_params.content["dt"]
            damping = step_params.content["damping"]
            metabolic_cost_speed = step_params.content["metabolic_cost_speed"]
            metabolic_cost_angular = step_params.content["metabolic_cost_angular"]
            x_max_arena = step_params.content["x_max_arena"]
            y_max_arena = step_params.content["y_max_arena"]

            key, *subkeys = random.split(agent.key, 5)

            # sample random movement
            forward_action = jax.random.uniform(subkeys[0], (), minval=0.0, maxval=1.0)
            angular_action = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=1.0)

            base_speed = (LINEAR_ACTION_OFFSET + LINEAR_ACTION_SCALE * forward_action) # depends on agent mass
            energy_factor = jnp.clip(energy[0]/50.0, 0.1, 1.0) # reduce speed based on available energy

            speed = base_speed * energy_factor * (1 + NOISE_SCALE * jax.random.normal(subkeys[2], ()))
            ang_speed = angular_action * energy_factor * (1 + NOISE_SCALE * jax.random.normal(subkeys[3], ()))

            x_new = jnp.clip(x + dt*x_dot, -x_max_arena, x_max_arena) # no wrap-around for now; may change it later
            y_new = jnp.clip(y + dt*y_dot, -y_max_arena, y_max_arena)
            ang_new = jnp.mod(ang + dt*ang_dot + jnp.pi, 2*jnp.pi) - jnp.pi

            x_dot_new = speed * jnp.cos(ang) - dt * x_dot * damping
            y_dot_new = speed * jnp.sin(ang) - dt * y_dot * damping
            ang_dot_new = ang_speed - dt * ang_dot * damping

            # speed-endurance tradeoff; energy consumption scales quadratically with speed
            metabolic_cost = metabolic_cost_speed * (jnp.abs(speed) / ACTION_SCALE) ** 2 + metabolic_cost_angular * jnp.abs(ang_speed) / ACTION_SCALE + BASIC_METABOLIC_COST
            # slow movement: energy efficient -> possible for longer periods of time
            # fast movement: uses more energy -> possible only for short periods of time
            energy_new = energy + energy_intake - metabolic_cost
            fitness_new = energy_new # for now: fitness is energy

            # reproduction: energy needs to be high enough (>threshold) for a certain amount of time + specific probability
            reproduction_prob = step_params.content["reproduction_prob"]
            reproduction_threshold = step_params.content["reproduction_threshold"]
            min_reproduction_time = step_params.content["min_reproduction_time"]
            current_timer = agent.state.content["reproduction_timer"]

            above_threshold = energy_new[0] >= reproduction_threshold
            new_timer = jax.lax.cond(
                above_threshold,
                lambda _: current_timer + dt, # increment timer if above threshold
                lambda _: 0.0, # reset timer if below threshold
                None
            )
            can_reproduce = new_timer >= min_reproduction_time

            key, reproduce_key = random.split(key)
            rand_float = jax.random.uniform(reproduce_key, shape=(1,))
            reproduce = jax.lax.cond(
                jnp.logical_and(can_reproduce, rand_float[0] < reproduction_prob),
                lambda _: 1,
                lambda _: 0, None
            )
            final_timer = jax.lax.cond(reproduce == 1, lambda _: 0.0, lambda _: new_timer, None) # reset timer if reproduction occurred

            # death: energy needs to be low enough (<=threshold) for a certain amount of time (min_death_time)
            death_threshold = step_params.content["death_threshold"]
            min_death_time = step_params.content["min_death_time"]
            current_death_timer = agent.state.content["death_timer"]

            below_threshold = energy_new[0] <= death_threshold
            new_death_timer = jax.lax.cond(
                below_threshold,
                lambda _: current_death_timer + dt,
                lambda _: 0.0,
                None
            )
            agent_is_dead = new_death_timer >= min_death_time

            new_state_content = {"x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new, "ang_dot": ang_dot_new,
                                 "energy": energy_new, "fitness": fitness_new, "reproduce": reproduce, "reproduction_timer": final_timer,
                                 "death_timer": new_death_timer}
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
                         "energy": energy, "fitness": fitness, "reproduce": 0, "reproduction_timer": 0.0,
                         "death_timer": 0.0}
        state = State(content=state_content)

        return agent.replace(state=state, age=0.0, active_state=1, key=key)

    @staticmethod
    def add_agent(agent, add_params): # reproduction; birth of a new agent
        parent_agent = add_params.content['agent_to_copy'] # from add_animals in ecosystem

        x = parent_agent.state.content["x"]
        y = parent_agent.state.content["y"]
        ang = parent_agent.state.content["ang"]
        energy = parent_agent.state.content["energy"]/2 # baby receives half the energy of parent
        fitness = parent_agent.state.content["fitness"]/2 # since fitness is the energy it also has to be halved

        x_dot = jnp.zeros((1,), dtype=jnp.float32)
        y_dot = jnp.zeros((1,), dtype=jnp.float32)
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)

        state_content = {"x": x, "y": y, "ang": ang, "x_dot": x_dot, "y_dot": y_dot, "ang_dot": ang_dot, "energy": energy, "fitness": fitness, "reproduce": 0, "reproduction_timer": 0, "death_timer": 0}
        state = State(content=state_content)

        params_content = {"radius": parent_agent.params.content["radius"],
                          "x_max": parent_agent.params.content["x_max"],
                          "y_max": parent_agent.params.content["y_max"],
                          "energy_begin_max": parent_agent.params.content["energy_begin_max"],
                          "mass": parent_agent.params.content["mass"],
                          "reproduction_prob": parent_agent.params.content["reproduction_prob"]
        }
        params = Params(content=params_content)
        return agent.replace(state=state, params=params, active_state=1, age=0.0)

    @staticmethod
    def half_energy(agent, set_params):
        state_content = {
            "x": agent.state.content["x"],
            "y": agent.state.content["y"],
            "ang": agent.state.content["ang"],
            "x_dot": agent.state.content["x_dot"],
            "y_dot": agent.state.content["y_dot"],
            "ang_dot": agent.state.content["ang_dot"],
            "energy": agent.state.content["energy"] / 2,  # parent loses half energy
            "fitness": agent.state.content["fitness"] / 2, # parent loses half fitness - q: reproduction limits agents survival for later episodes?
            "reproduce": 0,  # reset reproduce flag after reproduction
            "reproduction_timer": 0.0,  # reset reproduction timer
            "death_timer": agent.state.content["death_timer"]
        }
        state = State(content=state_content)
        return agent.replace(state=state)


def agent_interactions(sheep: Sheep, patches: Grass):

    def sheep_grass_interaction(one_sheep, patches):
        xs_patches = patches.state.content["x"] # shape (num_patches,)
        ys_patches = patches.state.content["y"]
        x_sheep = one_sheep.state.content["x"]
        y_sheep = one_sheep.state.content["y"]
        patches_radius = patches.params.content["radius"]

        distances = jnp.linalg.norm(jnp.stack((xs_patches - x_sheep, ys_patches - y_sheep), axis=1), axis=1).reshape(-1) # euclidean distances
        is_near_patch = jnp.where(distances < patches_radius, 1.0, 0.0)
        in_patches_num = jnp.sum(is_near_patch) # number of patches the sheep is in

        return in_patches_num, is_near_patch

    in_patches_nums, is_near_patch_matrix = jax.vmap(sheep_grass_interaction, in_axes=(0, None))(sheep, patches)
    is_energy_out_patches = jnp.any(is_near_patch_matrix, axis=1) # t/f if grass patch is being eaten by any sheep

    num_sheep_at_patch = jnp.maximum(jnp.sum(is_near_patch_matrix, axis=0), 1.0) # number of sheep present per grass patch (maximum; avoid dividing by 0)
    energy_sharing_matrix = jnp.divide(is_near_patch_matrix, num_sheep_at_patch)

    energy_intake_sheep = jnp.multiply(energy_sharing_matrix, patches.state.content["energy_offer"].reshape(-1))
    energy_intake_sheep = jnp.sum(energy_intake_sheep, axis=1).reshape(-1)

    #energy_lost_per_patch = jnp.multiply(is_energy_out_patches, patches.state.content["energy_offer"].reshape(-1))

    return is_energy_out_patches, energy_intake_sheep

jit_agent_interactions = jax.jit(agent_interactions)

@struct.dataclass
class PredatorPreyWorld:
    sheep_set: Set
    grass_set: Set

    @staticmethod
    def create_world(params, key):
        grass_params = params.content["grass_params"]
        sheep_params = params.content["sheep_params"]

        num_sheep = sheep_params["num_sheep"]

        key, sheep_key = random.split(key, 2)

        x_max_array = jnp.tile(jnp.array([sheep_params["x_max"]]), (num_sheep,))
        y_max_array = jnp.tile(jnp.array([sheep_params["y_max"]]), (num_sheep,))
        energy_begin_max_array = jnp.tile(jnp.array([sheep_params["energy_begin_max"]]), (num_sheep,))
        radius_array = jnp.tile(jnp.array([sheep_params["radius"]]), (num_sheep,))
        mass_array = jnp.tile(jnp.array([sheep_params["mass"]]), (num_sheep,))
        reproduction_prob_array = jnp.tile(jnp.array([sheep_params["reproduction_prob"]]), (num_sheep,))

        death_threshold_array = jnp.tile(jnp.array([sheep_params["death_threshold"]]), (num_sheep,))
        reproduction_threshold_array = jnp.tile(jnp.array([sheep_params["reproduction_threshold"]]), (num_sheep,))
        min_death_time_array = jnp.tile(jnp.array([sheep_params["min_death_time"]]), (num_sheep,))
        min_reproduction_time_array = jnp.tile(jnp.array([sheep_params["min_reproduction_time"]]), (num_sheep,))

        sheep_create_params = Params(content= {
            "x_max": x_max_array,
            "y_max": y_max_array,
            "energy_begin_max": energy_begin_max_array,
            "radius": radius_array,
            "mass": mass_array,
            "reproduction_prob": reproduction_prob_array,
            "death_threshold": death_threshold_array,
            "reproduction_threshold": reproduction_threshold_array,
            "min_death_time": min_death_time_array,
            "min_reproduction_time": min_reproduction_time_array
        })

        sheep = create_agents(agent=Sheep, params=sheep_create_params, num_agents=num_sheep, num_active_agents=num_sheep,
                              agent_type=sheep_params["agent_type"], key=sheep_key)

        sheep_set = Set(num_agents=num_sheep, num_active_agents=num_sheep, agents=sheep, id=0, set_type=sheep_params["agent_type"],
                        params=None, state=None, policy=None, key=None)

        key, grass_key = random.split(key, 2)

        num_grass = grass_params["num_grass"]
        x_max_array = jnp.tile(jnp.array([grass_params["x_max"]]), (num_grass,))
        y_max_array = jnp.tile(jnp.array([grass_params["y_max"]]), (num_grass,))
        energy_max_array = jnp.tile(jnp.array([grass_params["energy_stored_max"]]), (num_grass,))
        eat_rate_array = jnp.tile(jnp.array([grass_params["eat_rate"]]), (num_grass,))
        growth_rate_array = jnp.tile(jnp.array([grass_params["growth_rate"]]), (num_grass,))
        radius_array = jnp.tile(jnp.array([grass_params["radius"]]), (num_grass,))

        grass_create_params = Params(content= {"x_max": x_max_array,
                                               "y_max": y_max_array,
                                               "energy_stored_max": energy_max_array,
                                               "eat_rate": eat_rate_array,
                                               "growth_rate": growth_rate_array,
                                               "radius": radius_array,
        })

        grass = create_agents(agent=Grass, params=grass_create_params, num_agents=num_grass, num_active_agents=num_grass,
                              agent_type=grass_params["agent_type"], key=grass_key)

        grass_set = Set(num_agents=num_grass, num_active_agents=num_grass, agents=grass, id=1, set_type=grass_params["agent_type"],
                        params=None, state=None, policy=None, key=None)

        return PredatorPreyWorld(sheep_set=sheep_set, grass_set=grass_set)


def step_world(pp_world, _t):
    sheep_set = pp_world.sheep_set
    grass_set = pp_world.grass_set

    is_energy_out_patches, energy_intake_sheep = jit_agent_interactions(sheep_set.agents, grass_set.agents)

    grass_step_input = Signal(content={"is_energy_out": is_energy_out_patches})
    grass_step_params = Params(content={"dt": Dt})
    grass_set = jit_step_agents(Grass.step_agent, grass_step_params, grass_step_input, grass_set)

    sheep_step_input = Signal(content={"energy_intake": energy_intake_sheep})
    sheep_step_params = Params(content={"dt": Dt,
                                        "damping": DAMPING,
                                        "metabolic_cost_speed": METABOLIC_COST_SPEED,
                                        "metabolic_cost_angular": METABOLIC_COST_ANGULAR,
                                        "x_max_arena": MAX_SPAWN_X,
                                        "y_max_arena": MAX_SPAWN_Y,
                                        "action_scale": ACTION_SCALE,
    })
    sheep_set = jit_step_agents(Sheep.step_agent, sheep_step_params, sheep_step_input, sheep_set)

    render_data = Signal(content={"sheep_xs": sheep_set.agents.state.content["x"].reshape(-1, 1),
                                  "sheep_ys": sheep_set.agents.state.content["y"].reshape(-1, 1),
                                  "sheep_angles": sheep_set.agents.state.content["ang"].reshape(-1, 1),
                                  "grass_energies": grass_set.agents.state.content["energy"].reshape(-1, 1),
    })

    return pp_world.replace(sheep_set=sheep_set, grass_set=grass_set), render_data

jit_step_world = jax.jit(step_world)


def reset_world(pp_world):
    sheep_set_agents = pp_world.sheep_set.agents
    grass_set_agents = pp_world.grass_set.agents

    sheep_set_agents = jax.vmap(Sheep.reset_agent)(sheep_set_agents, None)
    grass_set_agents = jax.vmap(Grass.reset_agent)(grass_set_agents, None)

    sheep_set = pp_world.sheep_set.replace(agents=sheep_set_agents)
    grass_set = pp_world.grass_set.replace(agents=grass_set_agents)

    return pp_world.replace(sheep_set=sheep_set, grass_set=grass_set)

jit_reset_world = jax.jit(reset_world)



def scan_episode(pp_world: PredatorPreyWorld, ts):
    pass

def run_episode(pp_world: PredatorPreyWorld):
    pass

def main():
    pass

if __name__ == "__main__":
    main()
    print("simulation done")



















