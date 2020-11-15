'''
# =============================================================================
# In this file one can change the parameters as stated by the exercises. 
# Clarifications of the of the model parameters can be found in the assignment.
#
# Airport and canvas parameters:
# 	n_origin_airports = 20 [-]. Number of randomly generated origin airports.
# 	n_destination_airports = 20 [-]. Number of randomly generated destination airports.
# 	width = 750 [km]. Width of the canvas.
# 	height = 750 [km]. Height of the canvas.
# 	Boundaries of randomly generated airports:
# 	origin_airport_x: [0.01, 0.3]. 
# 	origin_airport_y: [0.01, 0.3].
# 	destination_airport_x: [0.7, 0.99].
# 	destination_airport_y: [0.7, 0.99].
#
# Flight parameters:
# 	n_flights = 50 [-]. Number of aircraft/flights.
# 	departure_window = 3 [s]. 
# 	speed = 0.3 [km/s]. Speed of the aircraft.
# 	communication_range = 200 [km]. Range 
# 	fuel_reduction = 0.75 [-]. When flying in formation, you use 75% of your original fuel consumption.
# 	negotiation_method = 0 [-]. Set which negotiation method to use 
#           (0: greedy algorithm, 1: CNP, 2: English, 3: Vickrey, 4: Japanese).
#
# Simulation parameters:
# 	n_iterations = 1 [-]. Number of simulation runs, used in the batch runner.

# =============================================================================
'''

from .metrics import *
import numpy as np


# This can be infinite, as the model should stop on its own when all agents have arrived at their destination.
max_steps = 10000 

# Multiple iterations are used when running the batchrunner.py:
n_iterations = 50

model_params = {
    "n_flights": 20,
    "n_origin_airports": 10,
    "n_destination_airports": 10,
    "communication_range": 100, #[km]
    "width": 300, # [km]
    "height": 300, # [km]
    "speed": 0.3, #[km / second]
    "max_speed": 0.500,
    "fuel_reduction": 0.75,
    "alliance_ratio": 0.40,
    #"manager_ratio": 0.40,
    "offer_ratio": 0.80,
    "entrance_fee": 50,
    "negotiation_method": 0,                # Greedy - 0 / CNP - 1 / English - 2 / Vickrey - 3 / Japanese - 4
    "joining_method": 1,                    # Complex - 0 / Middle Point - 1
    "bid_increase": 10,
    "departure_window": 3,
    "origin_airport_x": [0.01, 0.2],
    "origin_airport_y": [0.01, 0.2],
    "destination_airport_x": [0.7, 0.99], 
    "destination_airport_y": [0.7, 0.99],
}

# To run model with a variable parameter:
# example: variable_params = {"communication_range": [0, 100, 500]}
variable_params = {"manager_ratio": [0.50]}


model_reporter_parameters={"Total Fuel Used": compute_total_fuel_used, 
                           "steps": compute_model_steps, 
                           "new formations": new_formation_counter,
                           "Number of formations": formation_counter,
                           "Fraction of managers": manager_counter,
                           "added to formations": add_to_formation_counter, 
                           "Total planned Fuel": compute_planned_fuel,
                           "Total saved potential saved fuel": fuel_savings_closed_deals, 
                           "Real saved fuel": real_fuel_saved,
                           "Average delay": average_delay,
                           "Fuel saving ratio": fuel_saving_ratio,
                           "Alliance saving ratio": alliance_saving_ratio,
                           "Deal values": total_deal_value}

# In order to collect values like "deal-value", they should be specified on all agents.
agent_reporter_parameters={"Deal value": "deal_value"} 