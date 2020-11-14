# =============================================================================
# This file contains the function to do Contract Net Protocol (CNP).
# At the moment, assumptions are:
#       - there is a fixed minimum bid a manager will accept (should be replaced by a cost function)
#       - contractor bids expire at scheduled departure + delta_T
# =============================================================================
def do_CNP(flight):
    pass
    # if not flight.departure_time:
    #     raise Exception(
    #         "The object passed to the CNP has no departure time, therefore it seems that it is not a flight.")
    #
    # delta_T = 5  # Steps after which an offer expires (redundant in CNP, implementation for auctions)
    #
    # # Behaviour of a contractor (reaching out to the best possible call, saved fuel prioritised)
    # if not flight.manager and flight.formation_state == 0:
    #     targets = flight.find_CNP_candidate()
    #
    #     if targets:
    #         positive_savings = []
    #         for agent in targets:
    #             my_fuel_saved = flight.calculate_potential_fuelsavings(agent)
    #             their_fuel_saved = agent.calculate_potential_fuelsavings(flight)
    #             if my_fuel_saved > 0:
    #                 positive_savings.append({"agent": agent, "my_fuel_saved": my_fuel_saved, "their_fuel_saved": their_fuel_saved})
    #
    #         sorted_savings = sorted(positive_savings, key=lambda i: i["my_fuel_saved"], reverse=True)
    #         if positive_savings:
    #             best_offer = list(sorted_savings[0].values())
    #             flight.make_bid(best_offer[0], best_offer[1], best_offer[2], flight.alliance, flight.model.schedule.steps + delta_T)
    #
    #     elif not targets and flight.model.schedule.steps >= flight.departure_time + 50:
    #         flight.manager = True
    #         flight.accepting_bids = True
    #         flight.formation_merges = 0
    #
    # # Behaviour of a manager (accepting calls prioritizing small delay, if no candidates become contractor)
    # if flight.manager and flight.accepting_bids:
    #     if flight.received_bids:
    #         received_bids = flight.received_bids
    #         clean_bids = []
    #         for bid in received_bids:
    #             bid = list(bid.values())
    #             their_fuel = bid[1]
    #             my_fuel = bid[2]
    #             alliance = bid[3]
    #             delay = flight.calculate_delay(bid[0])
    #             score = ((their_fuel+my_fuel)*(1+alliance))/(delay+1)
    #             if score > 0:
    #                 clean_bids.append({"agent": bid[0], "their_fuel": bid[1], "our_fuel": bid[2], "score": score})
    #         if clean_bids:
    #             sorted_bids = sorted(clean_bids, key=lambda i: i["score"], reverse=True)
    #             winning_bid = list(sorted_bids[0].values())
    #             if (not flight.agents_in_my_formation) and (not winning_bid[0].agents_in_my_formation):
    #                 flight.start_formation(winning_bid[0], winning_bid[1], discard_received_bids=True)
    #             elif flight.agents_in_my_formation and flight.formation_role == "master" and (not winning_bid[0].agents_in_my_formation):
    #                 flight.add_to_formation(winning_bid[0], winning_bid[1], discard_received_bids=True)
    #             elif flight.agents_in_my_formation and flight.formation_role == "master" and winning_bid[0].agents_in_my_formation and winning_bid[0].formation_role == "master":
    #                 flight.add_to_formation(winning_bid[0], winning_bid[1], discard_received_bids=True)
    #         flight.received_bids = []
    #     elif len(
    #             flight.agents_in_my_formation) == 0 and flight.formation_state == 0 and flight.model.schedule.steps >= flight.departure_time + 50:
    #         neighbors = flight.find_CNP_candidate()
    #         if neighbors:
    #             flight.manager = False
    #             flight.accepting_bids = False
    #
    # # Behaviour of a manager in-formation (making calls to join other formations - merging)
    # if flight.formation_role == "master" and flight.formation_state == 2 and (not flight.received_bids) and flight.model.schedule.steps % 10 == 0:
    #     candidates = flight.find_formation_candidates()
    #     if candidates:
    #         positive_savings = []
    #         for agent in candidates:
    #             our_fuel_saved = flight.calculate_potential_fuelsavings(agent)
    #             their_fuel_saved = agent.calculate_potential_fuelsavings(flight)
    #             if our_fuel_saved > 0:
    #                 positive_savings.append({"agent": agent, "my_fuel_saved": our_fuel_saved, "their_fuel_saved": their_fuel_saved})
    #             sorted_savings = sorted(positive_savings, key=lambda i: i["my_fuel_saved"], reverse=True)
    #             if positive_savings:
    #                 best_offer = list(sorted_savings[0].values())
    #                 flight.make_bid(best_offer[0], best_offer[1], best_offer[2], flight.alliance, flight.model.schedule.steps + delta_T)
