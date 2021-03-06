'''
# =============================================================================
# Data functions are saved here (instead of Model.py) for a better overview. 
# These functions can be called upon by the DataCollector. 
# You can add more advanced metric here!
# =============================================================================
'''


def compute_total_fuel_used(model):
    return model.total_fuel_consumption


def compute_planned_fuel(model):
    return model.total_planned_fuel


def fuel_savings_closed_deals(model):
    return model.fuel_savings_closed_deals


def real_fuel_saved(model):
    return model.total_planned_fuel - model.total_fuel_consumption


def total_deal_value(model):
    deal_values = [agent.deal_value for agent in model.schedule.agents]
    return sum(deal_values)


def compute_total_flight_time(model):
    return model.total_flight_time

def average_delay(model):
    return model.total_delay/model.n_flights


def compute_model_steps(model):
    return model.schedule.steps


def new_formation_counter(model):
    return model.new_formation_counter


def add_to_formation_counter(model):
    return model.add_to_formation_counter


def formation_counter(model):
    return model.formation_counter


def manager_counter(model):
    return model.manager_counter / model.n_flights

def fuel_saving_ratio(model):
    return model.total_fuel_consumption/model.total_planned_fuel

def alliance_saving_ratio(model):
    return model.alliance_fuel_consumption/model.alliance_planned_fuel