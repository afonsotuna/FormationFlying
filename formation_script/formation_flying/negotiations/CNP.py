# =============================================================================
# This file contains the function to do Contract Net Protocol (CNP).
# At the moment, assumptions are:
#       - there is a fixed minimum bid a manager will accept (should be replaced by a cost function)
#       - contractor bids expire at scheduled departure + delta_T
# =============================================================================

def do_CNP(flight):
    if not flight.departure_time:
        raise Exception(
            "The object passed to the CNP has no departure time, therefore it seems that it is not a flight.")

    delta_T = 5  # Steps after which a bid expires

    # Behaviour of a contractor TODO make formation masters be able to bid
    if flight.auctioneer and flight.formation_state == 0:
        targets = flight.find_CNP_candidate()

        if targets:
            positive_savings = []
            for agent in targets:
                fuel_saved = flight.calculate_potential_fuelsavings(agent)
                joining_point = flight.find_joining_point(agent)
                dist_self = ((joining_point[0] - flight.pos[0]) ** 2 + (joining_point[1] - flight.pos[1]) ** 2) ** 0.5
                join_speed, their_speed = flight.calc_speed_to_point(agent)
                if int(dist_self) != 0:
                    time_to_join = dist_self / join_speed
                    if fuel_saved > 0:
                        positive_savings.append(
                            {"agent": agent, "fuel_saved": fuel_saved, "time_to_join": time_to_join})

            sorted(positive_savings, key=lambda i: i["fuel_saved"])
            if positive_savings:
                best_offer = list(positive_savings[0].values())
                flight.make_bid(best_offer[0], best_offer[1], best_offer[2], flight.model.schedule.steps + delta_T)

        elif not targets and flight.model.schedule.steps >= flight.departure_time + 50:
            flight.auctioneer = False
            flight.manager = True
            flight.accepting_bids = True

    # Behaviour of a manager (receiving bids or becoming a contractor)
    if flight.manager and flight.accepting_bids:
        if flight.received_bids:
            received_bids = flight.received_bids
            sorted(received_bids, key=lambda i: i["time_to_join"], reverse=True)
            for bid in received_bids:
                bid = list(bid.values())
                if flight.model.schedule.steps <= bid[3]:
                    if not flight.agents_in_my_formation:
                        flight.start_formation(bid[0], bid[1], discard_received_bids=True)
                    elif flight.agents_in_my_formation:
                        flight.add_to_formation(bid[0], bid[1], discard_received_bids=True)
                    break
        elif len(
                flight.agents_in_my_formation) == 0 and flight.formation_state == 0 and flight.model.schedule.steps >= flight.departure_time + 50:
            neighbors = flight.find_CNP_candidate()
            if neighbors:
                flight.manager = False
                flight.auctioneer = True
                flight.accepting_bids = False

    # Behaviour of a manager in-formation (bidding to join other formations)
    # if flight.manager and len(flight.agents_in_my_formation) > 0 and not flight.received_bids:
    #     targets = flight.find_CNP_candidate()
    #
    #     if targets:
    #         positive_savings = []
    #         for agent in targets:
    #             fuel_saved = flight.calculate_potential_fuelsavings(agent)
    #             joining_point = flight.find_joining_point(agent)
    #             dist_self = ((joining_point[0] - flight.pos[0]) ** 2 + (joining_point[1] - flight.pos[1]) ** 2) ** 0.5
    #             join_speed, their_speed = flight.calc_speed_to_point(agent)
    #             time_to_join = dist_self / join_speed
    #             if fuel_saved > 0:
    #                 positive_savings.append(
    #                     {"agent": agent, "fuel_saved": fuel_saved, "time_to_join": time_to_join})
    #             sorted(positive_savings, key=lambda i: i["fuel_saved"])
    #             if positive_savings:
    #                 best_offer = list(positive_savings[0].values())
    #                 flight.make_bid(best_offer[0], best_offer[1], best_offer[2], flight.model.schedule.steps + delta_T)
