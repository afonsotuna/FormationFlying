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
from..negotiations.english import do_English
from ..negotiations.japanese import do_Japanese
from ..negotiations.vickrey import do_Vickrey
import math


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

        self.planned_fuel = self.distance_between_points(self.pos, self.destination)
        self.planned_time = self.distance_between_points(self.pos, self.destination) / self.speed
        self.model.total_planned_fuel += self.planned_fuel

        self.fuel_consumption = 0  # A counter which counts the fuel consumed
        self.deal_value = 0  # All the fuel lost or won during bidding

        self.formation_state = 0  # 0 = no formation, 1 = committed, 2 = in formation, 3 = unavailable, 4 = adding to formation

        self.alliance = 0  # 0 = not in alliance, 1 = in alliance
        if self.model.random.random() < self.model.alliance_ratio:
            self.alliance = 1

        self.state = "scheduled"  # Can be scheduled, flying, or arrived

        self.last_bid_expiration_time = 0

        if self.alliance:
            self.model.alliance_planned_fuel += self.planned_fuel
        # =============================================================================
        #   Agents decide during initialization whether they are manager or auctioneer
        #   However, this can also be changed during the flight.
        #
        #   !!! TODO Exc. 1.3: implement when a manager can become an auctioneer and vice versa.!!!
        # =============================================================================
        self.accepting_bids = 0
        self.received_bids = []
        self.formation_role = False
        self.highest_bid = 0
        self.bid_made = False
        self.auction_on = False
        self.highest_bidder = 0
        self.auctioneer_target = 0

        if self.become_manager():
            self.model.manager_counter += 1
            self.manager = True
            self.auctioneer = True
            self.accepting_bids = True
        else:
            self.manager = False
            self.auctioneer = False
            self.accepting_bids = False

    # =============================================================================
    #   Defines if an agent should become a manager or not
    # =============================================================================

    # def become_manager(self):
    #     neighbors = self.model.space.get_neighbors(pos=self.pos, radius=self.communication_range, include_center=True)
    #     flight_neighbors = 0
    #     managers = 0
    #     contractors = 0
    #     for agent in neighbors:
    #         if type(agent) is Flight and agent != self:
    #             flight_neighbors += 1
    #             if agent.manager:
    #                 managers += 1
    #             else:
    #                 contractors += 1
    #     if managers == 0:
    #         be_manager = True
    #     elif managers / flight_neighbors >= 0.70:
    #         be_manager = False
    #     elif managers / flight_neighbors <= 0.30:
    #         be_manager = True
    #     else:
    #         if self.model.random.choice([0, 1]) == 1:
    #             be_manager = True
    #         else:
    #             be_manager = False
    #     return be_manager

    def become_manager(self):
        if self.model.random.random() >= self.model.manager_ratio:
            return False
        else:
            return True

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
            if self.model.negotiation_method == 2:
                do_English(self)
            if self.model.negotiation_method == 3:
                do_Vickrey(self)
            if self.model.negotiation_method == 4:
                do_Japanese(self)

    # =============================================================================
    #   This formula assumes that the route of both agents are of same length, 
    #   because joining- and leaving-points are taken to be as the middle-point 
    #   between their current positions / destinations.
    #
    #   !!! TODO Exc. 1.3: improve calculation joining/leaving point.!!!
    # =============================================================================

    def calculate_potential_fuelsavings(self, target_agent):
        # OWN SAVINGS
        if self.formation_state == 0:
            my_original_fuel = self.distance_between_points(self.pos, self.destination)
        elif self.formation_state == 2:
            fuel_to_leave = self.distance_between_points(self.pos, self.leaving_point)
            fuel_to_land = self.distance_between_points(self.leaving_point, self.destination)
            my_original_fuel = fuel_to_leave + fuel_to_land

        # Joining and leaving points
        joining_point = self.find_joining_point(target_agent)
        joining_speed, _ = self.calc_speed_to_joining_point(target_agent)
        if self.manager and self.agents_in_my_formation:
            leaving_point = self.leaving_point
        else:
            leaving_point = self.find_leaving_point(target_agent)

        my_fuel_to_join = self.distance_between_points(self.pos, joining_point)
        if not self.manager:
            my_formation_fuel = self.distance_between_points(joining_point, leaving_point) * self.model.fuel_reduction
        else:
            my_formation_fuel = self.distance_between_points(joining_point, leaving_point)
        my_approach_fuel = self.distance_between_points(leaving_point, self.destination)
        my_new_fuel = my_fuel_to_join + my_formation_fuel + my_approach_fuel

        savings = my_original_fuel - my_new_fuel

        # MY SLAVES' SAVINGS
        if self.formation_state == "master" and self.agents_in_my_formation:
            my_agents_total = 0
            for aircraft in self.agents_in_my_formation:
                aircraft_original_fuel = self.distance_between_points(aircraft.pos, self.destination)

                aircraft_fuel_to_join = self.distance_between_points(aircraft.pos, joining_point)
                aircraft_formation_fuel = self.distance_between_points(joining_point,
                                                                       leaving_point) * self.model.fuel_reduction
                aircraft_approach_fuel = self.distance_between_points(leaving_point, aircraft.destination)
                aircraft_new_fuel = aircraft_fuel_to_join + aircraft_formation_fuel + aircraft_approach_fuel

                my_agents_total += aircraft_original_fuel - aircraft_new_fuel
            savings += my_agents_total

        return savings

    # =============================================================================
    #   This formula calculated the delay incurred by aircraft A to join aircraft B
    #
    #   !!! TODO Review!!!
    # =============================================================================
    def calculate_delay(self, target_agent):
        time_from_position = self.distance_between_points(self.pos, self.destination) / self.speed
        joining_point = self.find_joining_point(target_agent)
        leaving_point = self.find_leaving_point(target_agent)
        my_meet_speed, _ = self.calc_speed_to_joining_point(target_agent)
        if my_meet_speed:
            time_to_meet = self.distance_between_points(self.pos, joining_point) / my_meet_speed
        else:
            time_to_meet = 0
        time_cruise = self.distance_between_points(joining_point, leaving_point) / self.speed
        time_approach = self.distance_between_points(leaving_point, self.destination) / self.speed
        time_if_formation = time_to_meet + time_cruise + time_approach
        delay = time_if_formation - time_from_position
        return delay

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

        elif len(self.agents_in_my_formation) > 0 and len(target_agent.agents_in_my_formation) == 0:
            self.model.agents_in_formation += 1
            self.accepting_bids = False

            if discard_received_bids:
                # Discard all bids that have been received
                self.received_bids = []

            # The joining point (common for both) and joining speeds are calculated
            self.joining_point = self.find_joining_point(target_agent)
            self.speed_to_joining, target_agent.speed_to_joining = self.calc_speed_to_joining_point(target_agent)

            # The list my_agents contains all agents in my formation, including myself
            my_agents = [self]
            for agent in self.agents_in_my_formation:
                my_agents.append(agent)  # These are the current formation agents

            # The target agent is appended to all my_agents members, and their status becomes joining formation
            for agent in my_agents:
                agent.agents_in_my_formation.append(target_agent)
                agent.formation_state = 4

            # if target_agent in my_agents:
            #     raise Exception("This is not correct")

            # In current implementation, all formation members (including myself) receive the same equal piece of bid
            bid_receivers = bid_value / (len(self.agents_in_my_formation) + 1)
            self.deal_value += bid_receivers
            for agent in self.agents_in_my_formation:
                agent.deal_value += bid_receivers

            target_agent.deal_value -= bid_value
            target_agent.formation_state = 1
            target_agent.agents_in_my_formation = my_agents[:]

            my_delay = self.calculate_delay(target_agent)
            self.model.total_delay += my_delay

            for agent in my_agents:
                agent.joining_point = self.joining_point
                agent.leaving_point = self.leaving_point
                agent.speed_to_joining = self.speed_to_joining
                agent.model.total_delay += my_delay
                agent.formation_state = 4

            target_agent.joining_point = self.joining_point
            target_agent.leaving_point = self.leaving_point
            target_agent.model.total_delay += target_agent.calculate_delay(self)

        elif len(target_agent.agents_in_my_formation) > 0 and len(self.agents_in_my_formation) > 0:
            self.model.formation_counter -= 1
            self.accepting_bids = False
            target_agent.accepting_bids = False

            if discard_received_bids:
                # Discard all bids that have been received
                self.received_bids = []
                target_agent.received_bids = []

            self.joining_point = self.find_joining_point(target_agent)
            self.speed_to_joining, target_agent.speed_to_joining = self.calc_speed_to_joining_point(target_agent)

            # Two lists, one for my own formation, one for the target agent's formation, both including ourselves
            my_agents = [self]
            their_agents = [target_agent]

            for agent in self.agents_in_my_formation:
                my_agents.append(agent)  # These are my current formation agents
            for agent in target_agent.agents_in_my_formation:
                their_agents.append(agent)  # These are their current formation agents

            # In current implementation, bid payment (and receipt) is divided equally amongst everyone
            bid_receivers = bid_value / (len(self.agents_in_my_formation) + 1)
            for agent in my_agents:
                agent.deal_value += bid_receivers

            bid_payers = bid_value / (len(target_agent.agents_in_my_formation) + 1)
            for agent in their_agents:
                agent.deal_value -= bid_payers

            for agent in my_agents:
                for target in their_agents:
                    agent.agents_in_my_formation.append(target)
                agent.formation_state = 4
            for agent in their_agents:
                for target in my_agents:
                    agent.agents_in_my_formation.append(target)
                agent.formation_state = 4

            my_delay = self.calculate_delay(target_agent)
            self.model.total_delay += my_delay

            for agent in my_agents:
                agent.joining_point = self.joining_point
                agent.leaving_point = self.leaving_point
                agent.speed_to_joining = self.speed_to_joining
                agent.model.total_delay += my_delay

            their_delay = target_agent.calculate_delay(self)
            target_agent.model.total_delay += their_delay

            for agent in their_agents:
                agent.joining_point = self.joining_point
                agent.leaving_point = self.leaving_point
                if not agent == target_agent:
                    agent.speed_to_joining = target_agent.speed_to_joining
                    agent.model.total_delay += their_delay

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

        self.model.formation_counter += 1
        self.model.fuel_savings_closed_deals += self.calculate_potential_fuelsavings(target_agent)
        self.deal_value += bid_value
        target_agent.deal_value -= bid_value

        self.model.total_delay += self.calculate_delay(target_agent)
        target_agent.model.total_delay += target_agent.calculate_delay(self)

        self.accepting_bids = False
        self.formation_role = "master"
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

            self.speed_to_joining, target_agent.speed_to_joining = self.calc_speed_to_joining_point(target_agent)

            target_agent.formation_state = 1
            self.formation_state = 1

        self.leaving_point = self.find_leaving_point(target_agent)
        self.agents_in_my_formation.append(target_agent)
        self.model.agents_in_formation += len(self.agents_in_my_formation) + 1
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
            if type(agent) is Flight and agent.accepting_bids:
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
                if agent.formation_state == 2 and agent.formation_role == "master" and agent.accepting_bids:
                    if not self == agent:
                        candidates.append(agent)
        return candidates

    def find_auction_candidates(self):
        neighbors = self.model.space.get_neighbors(pos=self.pos, radius=self.communication_range, include_center=True)
        candidates = []
        for agent in neighbors:
            if type(agent) is Flight and agent.accepting_bids:
                if not self == agent:
                    candidates.append(agent)
        return candidates

    # =========================================================================
    #   Making the bid.
    # =========================================================================
    def make_bid(self, bidding_target, my_fuel_saved, their_fuel_saved, alliance_status, bid_expiration_date):
        bid = {"bidding_agent": self, "my_fuel_saved": my_fuel_saved, "their_fuel_saved": their_fuel_saved,
               "alliance": alliance_status,
               "exp_date": bid_expiration_date}
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

    def distance_between_points(self, A, B):
        return ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5

    def kent_weights(self, n_agents):
        return -0.0017 * n_agents ** 3 + 0.0277 * n_agents ** 2 - 0.1639 * n_agents + 1.1357

    def three_point_circle(self, b, c, d):
        temp = c[0] ** 2 + c[1] ** 2
        bc = (b[0] ** 2 + b[1] ** 2 - temp) / 2
        cd = (temp - d[0] ** 2 - d[1] ** 2) / 2
        det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])
        if abs(det) < 1.0e-10:
            return None
        # Center
        cx = (bc * (c[1] - d[1]) - cd * (b[1] - c[1])) / det
        cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det
        # Radius
        radius = ((cx - b[0]) ** 2 + (cy - b[1]) ** 2) ** .5
        return [cx, cy], radius

    def find_joining_point(self, target_agent):
        if self.model.joining_method == 0:
            try:
                my_agents = len(self.agents_in_my_formation) + 1
                their_agents = len(target_agent.agents_in_my_formation) + 1
                if self.pos[1] > target_agent.pos[1]:
                    A = self.pos
                    B = target_agent.pos
                else:
                    B = self.pos
                    A = target_agent.pos
                AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
                w_a = self.kent_weights(my_agents)
                w_b = self.kent_weights(their_agents)
                w_c = self.kent_weights(my_agents + their_agents)
                AX = (AB / w_c) * w_b
                BX = (AB / w_c) * w_a
                alpha = math.acos(-(AX ** 2 - BX ** 2 - AB ** 2) / (2 * AX * AB))
                beta = (alpha / AX) * BX
                phi = math.atan((abs(B[1] - A[1])) / (abs(B[0] - A[0])))
                gamma = math.pi - beta - phi
                X = [A[0] - AX * math.cos(gamma), A[1] - AX * math.sin(gamma)]
                C = self.find_leaving_point(target_agent)

                centre, radius = self.three_point_circle(A, B, X)
                m = (C[1] - X[1]) / (C[0] - X[0])
                b = X[1] - m * X[0]
                coeff1 = 1 + m ** 2
                coeff2 = -2 * centre[0] + 2 * m * (b - centre[1])
                coeff3 = centre[0] ** 2 + (b - centre[1]) ** 2 - radius ** 2
                coeff = [coeff1, coeff2, coeff3]
                roots = np.roots(coeff)
                best_root = []
                for root in roots:
                    if len(best_root) == 0:
                        best_root.append(root)
                    elif root > best_root[0]:
                        best_root[0] = root
                x_P = best_root[0]
                y_P = m * x_P + b

                if x_P <= 0 or y_P <= 0 or x_P >= self.model.width or y_P >= self.model.height:
                    raise ValueError
                else:
                    return [x_P, y_P]

            except (ValueError, FloatingPointError, TypeError):
                x_P, y_P = self.calc_middle_point(self.pos, target_agent.pos)
                return [x_P, y_P]

        elif self.model.joining_method == 1:
            return self.calc_middle_point(self.pos, target_agent.pos)


    def find_leaving_point(self, target_agent):
        if self.model.joining_method == 0:
            try:
                my_agents = len(self.agents_in_my_formation) + 1
                their_agents = len(target_agent.agents_in_my_formation) + 1
                if self.destination[1] > target_agent.destination[1]:
                    A = self.destination
                    B = target_agent.destination
                else:
                    B = self.destination
                    A = target_agent.destination
                AB = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
                w_a = 1  # TODO make a separable formation at leaving
                w_b = 1
                w_c = self.kent_weights(my_agents + their_agents)
                AX = (AB / w_c) * w_b
                BX = (AB / w_c) * w_a
                alpha = math.acos(-(AX ** 2 - BX ** 2 - AB ** 2) / (2 * AX * AB))
                beta = (alpha / AX) * BX
                phi = math.atan((abs(B[1] - A[1])) / (abs(B[0] - A[0])))
                gamma = math.pi - beta - phi
                X = [A[0] - AX * math.cos(gamma), A[1] - AX * math.sin(gamma)]
                C = self.calc_middle_point(self.pos, target_agent.pos)

                centre, radius = self.three_point_circle(A, B, X)
                m = (C[1] - X[1]) / (C[0] - X[0])
                b = X[1] - m * X[0]
                coeff1 = 1 + m ** 2
                coeff2 = -2 * centre[0] + 2 * m * (b - centre[1])
                coeff3 = centre[0] ** 2 + (b - centre[1]) ** 2 - radius ** 2
                coeff = [coeff1, coeff2, coeff3]
                roots = np.roots(coeff)
                best_root = []
                for root in roots:
                    if len(best_root) == 0:
                        best_root.append(root)
                    elif root > best_root[0]:
                        best_root[0] = root
                x_P = best_root[0]
                y_P = m * x_P + b

                if x_P <= 0 or y_P <= 0 or x_P >= self.model.width or y_P >= self.model.height:
                    raise ValueError
                else:
                    return [x_P, y_P]

            except (ValueError, FloatingPointError, TypeError):
                x_P, y_P = self.calc_middle_point(self.destination, target_agent.destination)
                return [x_P, y_P]

        elif self.model.joining_method == 1:
            return self.calc_middle_point(self.destination, target_agent.destination)

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
                if self.formation_role == "master":
                    self.accepting_bids = True
                else:
                    self.accepting_bids = False

        if self.state == "flying":
            self.model.total_flight_time += 1
            if self.formation_state == 2:
                if self.formation_role != "master":
                    # If in formation, and not the master, fuel consumption is 75% of normal fuel consumption.
                    f_c = self.speed * self.model.fuel_reduction
                elif self.formation_role == "master":
                    f_c = self.speed
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
            if self.alliance:
                self.model.alliance_fuel_consumption += f_c

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
    def calc_speed_to_joining_point(self, target_agent):

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
