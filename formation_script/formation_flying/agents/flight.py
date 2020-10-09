'''
# =============================================================================
#    In this file the Flight-styleagent is defined.
#
#    Flights have a communication_range that defines the radius in which they 
#    look for their neighbors to negotiate with. They negotiate who to form a 
#    formation with in order to save fuel.  
#    
#    Different negotiation methods can be applied. In the parameter files one 
#    can set 'negototiation_method' which defines which method will be used. 
#    The base model only includes the greedy algorithm.
#
# =============================================================================
'''

import numpy as np

from mesa import Agent
from .airports import Airport
from ..negotiations.greedy import do_greedy  # !!! Don't forget the others.
from ..negotiations.CNP import do_CNP
import math


def calc_distance(p1, p2):
    # p1 = tuple(p1)
    # p2 = tuple(p2)
    dist = (((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
    return dist


class Flight(Agent):

    # =========================================================================
    # Create a new Flight agent.
    #
    # Args:
    #     unique_id: Unique agent identifier.
    #     pos: Starting position
    #     destination: the position of the destination
    #     destination_agent: the agent of the destination airport
    #     speed: Distance to move per step.
    #     departure_time: step of the model at which the flight should depart its origin
    #
    #     heading: numpy vector for the Flight's direction of movement.
    #     communication_range: Radius to look around for Flights to negotiate with.
    # =========================================================================

    def __init__(
            self,
            unique_id,
            model,
            pos,
            destination_agent,
            destination_pos,
            departure_time,
            speed,
            max_speed,
            communication_range,
    ):

        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.destination = np.array(destination_pos)
        self.destination_agent = destination_agent
        self.speed = speed
        self.max_speed = max_speed
        self.departure_time = departure_time
        self.heading = [self.destination[0] - self.pos[0], self.destination[1] - self.pos[1]]
        self.communication_range = communication_range

        # =====================================================================
        #   Initialize parameters, the values will not be used later on.
        # =====================================================================
        self.agents_in_my_formation = []

        self.leaving_point = [-10, -10]
        self.joining_point = [-10, -10]

        self.planned_fuel = calc_distance(self.pos, self.destination)
        self.model.total_planned_fuel += self.planned_fuel

        self.fuel_consumption = 0  # A counter which counts the fuel consumed
        self.deal_value = 0  # All the fuel lost or won during bidding

        self.formation_state = 0  # 0 = no formation, 1 = committed, 2 = in formation, 3 = unavailable, 4 = adding to formation

        self.state = "scheduled"  # Can be scheduled, flying, or arrived

        self.last_bid_expiration_time = 0

        # =============================================================================
        #   Agents decide during initialization whether they are manager or auctioneer
        #   However, this can also be changed during the flight.
        #
        #   !!! TODO Exc. 1.3: implement when a manager can become an auctioneer and vice versa.!!!
        # =============================================================================
        self.accepting_bids = 0
        self.received_bids = []

        if self.become_manager():
            self.model.manager_counter += 1
            self.manager = True
            self.auctioneer = False
            self.accepting_bids = 1
            self.formation_merges = 0
        else:
            self.manager = False
            self.auctioneer = True
            self.accepting_bids = 0

    # =============================================================================
    #   Defines if an agent should become a manager or not
    # =============================================================================
    def become_manager(self):
        neighbors = self.model.space.get_neighbors(pos=self.pos, radius=self.communication_range, include_center=True)
        flight_neighbors = 0
        managers = 0
        contractors = 0
        for agent in neighbors:
            if type(agent) is Flight and agent != self:
                flight_neighbors += 1
                if agent.manager:
                    managers += 1
                else:
                    contractors += 1
        if managers == 0:
            be_manager = True
        elif managers / flight_neighbors >= 0.40:
            be_manager = False
        elif managers / flight_neighbors <= 0.15:
            be_manager = True
        else:
            if self.model.random.choice([0, 1]) == 1:
                be_manager = True
            else:
                be_manager = False
        return be_manager

    # =============================================================================
    #   In advance, the agent moves (physically) to the next step (after having negotiated)
    # =============================================================================
    def advance(self):
        self.do_move()

    # =============================================================================
    #   In the "step", the negotiations are performed.
    #   Check if the agent is flying, because negotiations are only allowed in the air.
    #
    #   !!! TODO Exc. 2: implement other negotiation methods.!!!
    # =============================================================================
    def step(self):
        if self.state == "flying":
            if self.model.negotiation_method == 0:
                do_greedy(self)

            if len(self.agents_in_my_formation) > 0 and self.formation_state == 0:
                raise Exception(
                    "Agent status is no-formation, but it has agents registered as being in its formation...")

            if self.model.negotiation_method == 1:
                do_CNP(self)
            # if self.model.negotiation_method == 2:
            #     do_English(self)
            # if self.model.negotiation_method == 3:
            #     do_Vickrey(self)
            # if self.model.negotiation_method == 4:
            #     do_Japanese(self)

    # =============================================================================
    #   This formula assumes that the route of both agents are of same length, 
    #   because joining- and leaving-points are taken to be as the middle-point 
    #   between their current positions / destinations.
    #
    #   !!! TODO Exc. 1.3: improve calculation joining/leaving point.!!!
    # =============================================================================
    def calculate_potential_fuelsavings(self, target_agent):
        if len(self.agents_in_my_formation) == 0 and len(target_agent.agents_in_my_formation) == 0:
            joining_point = self.find_joining_point(target_agent)
            leaving_point = self.calc_middle_point(self.destination, target_agent.destination)

            original_distance = calc_distance(self.pos, self.destination) + calc_distance(target_agent.pos,
                                                                                          target_agent.destination)

            added_distance_agent1 = calc_distance(self.pos, joining_point) + calc_distance(leaving_point,
                                                                                           self.destination)
            added_distance_agent2 = calc_distance(target_agent.pos, joining_point) + calc_distance(
                target_agent.destination, leaving_point)
            formation_distance = calc_distance(leaving_point, joining_point) * 2

            new_total_distance = self.model.fuel_reduction * formation_distance + added_distance_agent1 + added_distance_agent2

            fuel_savings = original_distance - new_total_distance

        else:
            if len(self.agents_in_my_formation) > 0 and len(target_agent.agents_in_my_formation) > 0:
                if len(self.agents_in_my_formation) > len(target_agent.agents_in_my_formation):
                    formation_leader = self
                    formation_joiner = target_agent
                elif len(self.agents_in_my_formation) < len(target_agent.agents_in_my_formation):
                    formation_leader = target_agent
                    formation_joiner = self
                elif self.planned_fuel > target_agent.planned_fuel:
                    formation_leader = self
                    formation_joiner = target_agent
                elif self.planned_fuel < target_agent.planned_fuel:
                    formation_leader = target_agent
                    formation_joiner = self
                else:
                    if self.model.random.choice([0, 1]) == 1:
                        formation_leader = self
                        formation_joiner = target_agent
                    else:
                        formation_leader = target_agent
                        formation_joiner = self
                n_agents_in_formation = len(self.agents_in_my_formation) + len(target_agent.agents_in_my_formation) + 1

            elif len(self.agents_in_my_formation) > 0 and len(target_agent.agents_in_my_formation) == 0:
                formation_leader = self
                formation_joiner = target_agent
                n_agents_in_formation = len(self.agents_in_my_formation) + 1

            elif len(self.agents_in_my_formation) == 0 and len(target_agent.agents_in_my_formation) > 0:
                formation_leader = target_agent
                formation_joiner = self
                n_agents_in_formation = len(target_agent.agents_in_my_formation) + 1

            joining_point = self.find_joining_point(target_agent)
            leaving_point = formation_leader.leaving_point

            # Fuel for leader
            new_distance_formation = calc_distance(formation_leader.pos, joining_point) + calc_distance(joining_point,
                                                                                                        leaving_point)
            total_fuel_formation = self.model.fuel_reduction * n_agents_in_formation * new_distance_formation

            original_distance_formation = calc_distance(formation_leader.pos, leaving_point)
            original_fuel_formation = self.model.fuel_reduction * n_agents_in_formation * original_distance_formation

            fuel_savings_formation = original_fuel_formation - total_fuel_formation

            # Fuel for new agent
            fuel_to_joining_joiner = calc_distance(self.pos, joining_point)
            fuel_in_formation_joiner = calc_distance(joining_point, leaving_point) * self.model.fuel_reduction
            fuel_from_leaving_joiner = calc_distance(leaving_point, formation_joiner.destination)
            total_fuel_joiner = fuel_to_joining_joiner + fuel_in_formation_joiner + fuel_from_leaving_joiner

            original_fuel_joiner = calc_distance(formation_joiner.pos, formation_joiner.destination)

            fuel_savings_joiner = original_fuel_joiner - total_fuel_joiner

            fuel_savings = fuel_savings_joiner + fuel_savings_formation

        return fuel_savings

    # =========================================================================
    #   Add the chosen flight to the formation. While flying to the joining point 
    #   of a new formation, managers temporarily don't accept any new bids.
    #
    #   Calculate how the "bid_value" is divided.
    #   The agents already in the formation, share the profit from the bid equally.
    #
    #   !!! TODO Exc. 1.3: improve calculation joining/leaving point.!!!
    # =========================================================================
    def add_to_formation(self, target_agent, bid_value, discard_received_bids=True):
        self.model.fuel_savings_closed_deals += self.calculate_potential_fuelsavings(target_agent)

        if len(target_agent.agents_in_my_formation) > 0 and len(self.agents_in_my_formation) == 0:
            raise Exception("Model isn't designed for this scenario.")

        if len(self.agents_in_my_formation) > 0 and len(target_agent.agents_in_my_formation) == 0:
            self.model.add_to_formation_counter += 1
            self.accepting_bids = False

            if discard_received_bids:
                # Discard all bids that have been received
                self.received_bids = []

            self.joining_point = self.find_joining_point(target_agent)
            self.speed_to_joining, target_agent.speed_to_joining = self.calc_speed_to_point(target_agent)

            my_agents = [self]
            for agent in self.agents_in_my_formation:
                my_agents.append(agent)  # These are the current formation agents

            for agent in my_agents:
                agent.agents_in_my_formation.append(target_agent)
                agent.formation_state = 4

            if target_agent in my_agents:
                raise Exception("This is not correct")

            bid_receivers = bid_value / (len(
                self.agents_in_my_formation) + 1)

            self.deal_value += bid_receivers
            for agent in self.agents_in_my_formation:
                agent.deal_value += bid_receivers

            target_agent.deal_value -= bid_value

            target_agent.formation_state = 1

            target_agent.agents_in_my_formation = my_agents

            for agent in my_agents:
                agent.joining_point = self.joining_point
                agent.leaving_point = self.leaving_point
                agent.speed_to_joining = self.speed_to_joining

            target_agent.joining_point = self.joining_point
            target_agent.leaving_point = self.leaving_point

        if len(target_agent.agents_in_my_formation) > 0 and len(self.agents_in_my_formation) > 0:
            self.model.add_to_formation_counter += 1
            self.accepting_bids = False

            if discard_received_bids:
                # Discard all bids that have been received
                self.received_bids = []

            self.joining_point = self.find_joining_point(target_agent)
            self.speed_to_joining, target_agent.speed_to_joining = self.calc_speed_to_point(target_agent)

            my_agents = [self]
            their_agents = [target_agent]

            for agent in self.agents_in_my_formation:
                my_agents.append(agent)     # These are the current formation agents
            for agent in target_agent.agents_in_my_formation:
                their_agents.append(agent)  # These are their current formation agents

            bid_receivers = bid_value / (len(self.agents_in_my_formation) + 1)
            for agent in my_agents:
                agent.deal_value += bid_receivers

            bid_payers = bid_value / (len(target_agent.agents_in_my_formation) + 1)
            for agent in their_agents:
                agent.deal_value -= bid_payers

            for agent in my_agents:
                agent.agents_in_my_formation.append(agent)
                agent.formation_state = 4
            for agent in their_agents:
                agent.agents_in_my_formation.append(agent)
                agent.formation_state = 4


            for agent in my_agents:
                agent.joining_point = self.joining_point
                agent.leaving_point = self.leaving_point
                agent.speed_to_joining = self.speed_to_joining

            for agent in their_agents:
                agent.joining_point = self.joining_point
                agent.leaving_point = self.leaving_point
                if not agent == target_agent:
                    agent.speed_to_joining = target_agent.speed_to_joining

            target_agent.formation_role = "slave"

    # =========================================================================
    #   The value of the bid is added to the "deal value" of the manager, 
    #   and removed from the auctioneer. A manager leads the formation, the rest
    #   are 'slaves' to the manager.
    #
    #   !!! TODO Exc. 1.3: improve calculation joining/leaving point.!!!
    # =========================================================================
    def start_formation(self, target_agent, bid_value, discard_received_bids=True):
        if self == target_agent:
            raise Exception("ERROR: Trying to start a formation with itself")
        if len(self.agents_in_my_formation) > 0 or len(target_agent.agents_in_my_formation) > 0:
            raise Exception("Starting a formation with an agent that is already in a formation!")

        self.model.new_formation_counter += 1
        self.model.fuel_savings_closed_deals += self.calculate_potential_fuelsavings(target_agent)
        self.deal_value += bid_value
        target_agent.deal_value -= bid_value

        self.accepting_bids = False
        self.formation_role = "manager"
        target_agent.formation_role = "slave"

        # You can use the following error message if you want to ensure that managers can only start formations with
        # auctioneers. The code itself has no functionality, but is a "check"

        # if not self.manager and target_agent.auctioneer:
        #   raise Exception("Something is going wrong")

        if discard_received_bids:
            self.received_bids = []

        if self.distance_to_destination(target_agent.pos) < 0.001:
            # Edge case where agents are at the same spot.
            self.formation_state = 2
            target_agent.formation_state = 2
            self.accepting_bids = True

        else:
            self.joining_point = self.find_joining_point(target_agent)

            target_agent.joining_point = self.joining_point
            self.speed_to_joining, target_agent.speed_to_joining = self.calc_speed_to_point(target_agent)

            target_agent.formation_state = 1
            self.formation_state = 1

        self.leaving_point = self.calc_middle_point(self.destination, target_agent.destination)
        self.agents_in_my_formation.append(target_agent)
        target_agent.agents_in_my_formation.append(self)
        target_agent.leaving_point = self.leaving_point

    # =============================================================================
    #   This function finds the agents to make a bid to, and returns a list of these agents.
    #   In the current implementation, it randomly returns an agent,
    #   instead of deciding which manager it wants tomake a bid to.
    # =============================================================================

    def find_greedy_candidate(self):
        neighbors = self.model.space.get_neighbors(pos=self.pos, radius=self.communication_range, include_center=True)
        candidates = []
        for agent in neighbors:
            if type(agent) is Flight:
                if agent.formation_state == 0 or agent.formation_state == 2:
                    if not self == agent:
                        # Pass if it is the current agent
                        candidates.append(agent)
        return candidates

    def find_CNP_candidate(self):
        neighbors = self.model.space.get_neighbors(pos=self.pos, radius=self.communication_range, include_center=True)
        candidates = []
        for agent in neighbors:
            if type(agent) is Flight:
                if (agent.formation_state == 0 and agent.manager) or (
                        agent.formation_state == 2 and agent.formation_role == "master"):
                    if not self == agent:
                        candidates.append(agent)
        return candidates


    def find_formation_candidates(self):
        neighbors = self.model.space.get_neighbors(pos=self.pos, radius=self.communication_range, include_center=True)
        candidates = []
        for agent in neighbors:
            if type(agent) is Flight:
                if agent.formation_state == 2: #and agent.formation_role == "master":
                    if not self == agent:
                        candidates.append(agent)
        return candidates

    # =========================================================================
    #   Making the bid.
    # =========================================================================
    def make_bid(self, bidding_target, fuel_saved, time_to_join, bid_expiration_date):
        bid = {"bidding_agent": self, "fuel_saved": fuel_saved, "time_to_join": time_to_join, "exp_date": bid_expiration_date}
        bidding_target.received_bids.append(bid)

    # =========================================================================
    #   This function randomly chooses a new destination airport. 
    #
    #   !!! This can be used if you decide to close airports on the fly while 
    #   implementing de-commitment (bonus-assignment).!!!
    # =========================================================================
    def find_new_destination(self):

        open_destinations = []
        for agent in self.model.schedule.agents:
            if type(agent) is Airport:
                if agent.airport_type == "Destination":
                    open_destinations.append(agent)

        self.destination_agent = self.model.random.choice(open_destinations)
        self.destination = self.destination_agent.pos

        # You could add code here to decommit from the current bid.

    # =========================================================================
    #   'calc_middle_point'
    #   Calculates the middle point between two geometric points a & b. 
    #   Is used to calculate the joining- and leaving-points of a formation.
    #
    #   'distance_to_destination' 
    #   Calculates the distance to one point (destination) from an agents' current point.
    #
    #   !!! TODO Exc. 1.3: improve calculation joining/leaving point.!!!
    # =========================================================================
    def calc_middle_point(self, a, b):
        return [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]

    def distance_to_destination(self, destination):
        # 
        return ((destination[0] - self.pos[0]) ** 2 + (destination[1] - self.pos[1]) ** 2) ** 0.5

    def find_joining_point(self, target_agent):
        departure_point = self.calc_middle_point(self.pos, target_agent.pos)
        arrival_point = self.calc_middle_point(self.destination, target_agent.destination)
        m = (arrival_point[1] - departure_point[1]) / (arrival_point[0] - departure_point[0])
        b = departure_point[1] - m * departure_point[0]
        if self.pos[0] >= target_agent.pos[0]:
            joining_point = [self.pos[0], m * self.pos[0] + b]
        else:
            joining_point = [target_agent.pos[0], m * target_agent.pos[0] + b]
        return joining_point

    # =========================================================================
    #   This function actually moves the agent. It considers many different 
    #   scenarios in the if's and elif's, which are explained step-by-step.
    # =========================================================================
    def do_move(self):

        if self.distance_to_destination(self.destination) <= self.speed / 2:
            # If the agent is within reach of its destination, the state is changed to "arrived"
            self.state = "arrived"

        elif self.model.schedule.steps >= self.departure_time:
            # The agent only starts flying if it is at or past its departure time.
            self.state = "flying"

            if self.formation_state == 2 and self.distance_to_destination(self.leaving_point) <= self.speed / 2:
                # If agent is in formation & close to leaving-point, leave the formation
                self.state = "flying"
                self.formation_state = 0
                self.agents_in_my_formation = []

            if (self.formation_state == 1 or self.formation_state == 4) and \
                    self.distance_to_destination(self.joining_point) <= self.speed_to_joining / 2:
                # If the agent reached the joining point of a new formation, 
                # change status to "in formation" and start accepting new bids again.
                self.formation_state = 2
                self.accepting_bids = True

        if self.state == "flying":
            self.model.total_flight_time += 1
            if self.formation_state == 2 and self.formation_role != "master":
                # If in formation, and not the master, fuel consumption is 75% of normal fuel consumption.
                f_c = self.model.fuel_reduction * self.speed
                self.heading = [self.leaving_point[0] - self.pos[0], self.leaving_point[1] - self.pos[1]]
                self.heading /= np.linalg.norm(self.heading)
                new_pos = self.pos + self.heading * self.speed



            elif self.formation_state == 1 or self.formation_state == 4:
                # While on its way to join a new formation
                if self.formation_state == 4 and len(
                        self.agents_in_my_formation) > 0 and self.formation_role != "master":
                    f_c = self.speed_to_joining * self.model.fuel_reduction
                else:
                    f_c = self.speed_to_joining

                self.heading = [self.joining_point[0] - self.pos[0], self.joining_point[1] - self.pos[1]]
                self.heading /= np.linalg.norm(self.heading)
                new_pos = self.pos + self.heading * self.speed_to_joining

            else:
                self.heading = [self.destination[0] - self.pos[0], self.destination[1] - self.pos[1]]
                f_c = self.speed
                self.heading /= np.linalg.norm(self.heading)
                new_pos = self.pos + self.heading * self.speed

            if f_c < 0:
                raise Exception("Fuel cost lower than 0")

            self.model.total_fuel_consumption += f_c
            self.fuel_consumption += f_c

            self.model.space.move_agent(self, new_pos)

    def is_destination_open(self):
        if self.destination_agent.airport_type == "Closed":
            return False
        else:
            return True

    # =========================================================================
    #   Calculates the speed to joining point.
    #
    #   !!! TODO Exc. 1.3: improve calculation joining/leaving point.!!!
    # =========================================================================
    def calc_speed_to_point(self, target_agent):

        joining_point = self.find_joining_point(target_agent)
        dist_self = ((joining_point[0] - self.pos[0]) ** 2 + (joining_point[1] - self.pos[1]) ** 2) ** 0.5
        dist_target = ((joining_point[0] - target_agent.pos[0]) ** 2 + (
                joining_point[1] - target_agent.pos[1]) ** 2) ** 0.5

        if dist_self >= dist_target:
            own_speed = self.max_speed
            rest = dist_self % own_speed
            regular_time = math.floor(dist_self / own_speed)
            if rest > 0:
                time = regular_time + 1
            elif rest == 0:
                time = regular_time
            if time == 0:
                time = 0.0001
            target_speed = dist_target / time
        else:
            target_speed = target_agent.max_speed
            rest = dist_target % target_speed
            regular_time = math.floor(dist_target / target_speed)
            if rest > 0:
                time = regular_time + 1
            elif rest == 0:
                time = regular_time
            own_speed = dist_self / time

        return own_speed, target_speed
