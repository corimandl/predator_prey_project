"""
First version of AB predator-prey simulation

Characteristics:
- One species only (prey; sheep)
- One resource; grass patches
- Agents move randomly (both linear and angular)
- Energy/Mass/Speed trade-offs
"""

from structs import *
from functions import *

import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct

from train_v7 import LINEAR_ACTION_SCALE

#from evosax import CMA_ES

MAX_WORLD_X = 10000.0
MAX_WORLD_Y = 10000.0

MAX_SPAWN_X = 500.0
MAX_SPAWN_Y = 500.0

Dt = 0.1 # discrete time increments
KEY = jax.random.PRNGKey(0) # which seed?

# Sheep parameters
NUM_SHEEP = 100
SHEEP_RADIUS = 5.0
SHEEP_ENERGY_BEGIN_MAX = 50.0
SHEEP_MASS_BEGIN = 5.0 # initial sheep mass at birth
SHEEP_AGENT_TYPE = 1

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
NUM_ACTIONS = 2
ACTION_SCALE = 1.0
LINEAR_ACTION_SCALE = ACTION_SCALE * SHEEP_RADIUS
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
                                                    "num_sheep": NUM_SHEEP
                                                    },
                                   "grass_params": {"x_max": MAX_SPAWN_X,
                                                    "y_max": MAX_SPAWN_Y,
                                                    "energy_stored_max": ENERGY_STORED_MAX,
                                                    "growth_rate": GROWTH_RATE,
                                                    "eat_rate": EAT_RATE,
                                                    "radius": GRASS_RADIUS,
                                                    "agent_type": GRASS_AGENT_TYPE,
                                                    "num_grass": NUM_GRASS
                                                    },
                                   "action_params": {"num_actions" : NUM_ACTIONS}
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

        return Grass(id=id, state=state, params=params, active_state=active_state, agent_type=type, age = 0.0, key=key) # policy = None ?

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

    












