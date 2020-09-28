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

    min_bid = 10
    # Behaviour of a manager
    if flight.manager:
        if flight.accepting_bids:
            if flight.received_bids:
                for bid in flight.received_bids:
                    bid = list(bid.values())
                    if bid[1] > min_bid and bid[2] <= flight.model.schedule.steps:
                        if len(flight.agents_in_my_formation) == 0:
                            flight.start_formation(bid[0], bid[1], discard_received_bids=False)
                        else:
                            flight.add_to_formation(bid[0], bid[1], discard_received_bids=False)


    delta_T = 10
    # Behaviour of a contractor
    if flight.auctioneer and flight.formation_state == 0:
        neighbors = flight.find_candidate()
        saving_per_neighbor = []
        for manager in neighbors:
            savings = flight.calculate_potential_fuelsavings(manager)
            saving_per_neighbor.append({"agent": manager, "saving": savings})
        sorted(saving_per_neighbor, key=lambda i: i["saving"])
        best_offer = list(saving_per_neighbor[0].values())
        print(best_offer)
        flight.make_bid(best_offer[0], best_offer[1], flight.departure_time+delta_T)
