'''
# =============================================================================
# This file contains the function to do a Vickrey auction. 
# =============================================================================
'''


def do_Vickrey(flight):
    delta_T = 5

    # AUCTIONEER

    if flight.auctioneer and flight.accepting_bids:
        if flight.received_bids:
            if len(flight.received_bids) > 1:
                for bid in flight.received_bids:
                    current_bid = list(bid.values())
                    delay_time = flight.calculate_delay(current_bid[0])
                    score = 500 * current_bid[3] - delay_time * 30 + current_bid[1] * 2
                    bid["score"] = score
                sorted_bids = sorted(flight.received_bids, key=lambda i: i["score"], reverse=True)
                flight.received_bids.remove(sorted_bids[0])
                winner = list(sorted_bids[0].values())[0]
                price = list(sorted_bids[1].values())[1]
                for bid in flight.received_bids:
                    current_bid = list(bid.values())
                    current_bid[0].bid_made = False
            if len(flight.received_bids) == 1:
                winner = list(flight.received_bids[0].values())[0]
                price = list(flight.received_bids[0].values())[1]
            if (not flight.agents_in_my_formation) and (not winner.agents_in_my_formation):
                flight.start_formation(winner, price, discard_received_bids=True)
            elif flight.agents_in_my_formation and flight.formation_role == "master" and (
                    not winner.agents_in_my_formation):
                flight.add_to_formation(winner, price, discard_received_bids=True)
            elif flight.agents_in_my_formation and flight.formation_role == "master" and winner.agents_in_my_formation and winner.formation_role == "master":
                flight.add_to_formation(winner, price, discard_received_bids=True)

    # BIDDER

    # Open bid
    if not flight.auctioneer and flight.formation_state == 0 and flight.bid_made is False:
        candidates = flight.find_auction_candidates()
        scores = []
        bid_ratio = 0.4
        if candidates:
            for target_agent in candidates:
                alliance_status = target_agent.alliance
                delay_time = flight.calculate_delay(target_agent)
                fuel_saved = flight.calculate_potential_fuelsavings(target_agent)
                bid = bid_ratio * fuel_saved
                score = 500 * alliance_status - delay_time + fuel_saved
                scores.append({"agent": target_agent, "delay": delay_time, "alliance": alliance_status, "score": score,
                               "bid": bid})
            sorted_scores = sorted(scores, key=lambda i: i["score"], reverse=True)
            best_option = list(sorted_scores[0].values())
            agent = best_option[0]
            delay_time = best_option[1]
            alliance_status = best_option[2]
            bid = best_option[4]
            flight.make_bid(agent, bid, delay_time, alliance_status, flight.model.schedule.steps + delta_T)
            flight.auctioneer_target = agent
            flight.bid_made = True
