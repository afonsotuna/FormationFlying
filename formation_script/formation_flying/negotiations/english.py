'''
# =============================================================================
# This file contains the function to do an English auction. 
# =============================================================================
'''

def do_English(flight):

    delta_T = 5

    # AUCTIONEER

    if flight.auctioneer and flight.accepting_bids:
        winner = False

        if not flight.auction_on and flight.received_bids:
            start_total = 0
            for bid in flight.received_bids:
                current_bid = list(bid.values())
                net_result = current_bid[1] + current_bid [2]
                start_total += net_result
                bid["score"] = net_result
            flight.highest_bid = start_total/len(flight.received_bids)
            flight.turn_count = 1
            flight.accepting_bids = False
            flight.auction_on = True

        if flight.auction_on and len(flight.received_bids) > 1:
            sorted_bids = sorted(flight.received_bids, key=lambda i: i["score"], reverse=True)
            starting_bid = flight.highest_bid
            winner = list(sorted_bids[0].values())
            second_highest = list(sorted_bids[1].values())[5]
            price_to_pay = (int(second_highest/flight.model.bid_increase)+1) * flight.model.bid_increase
            flight.turn_count += int(price_to_pay-starting_bid) / flight.model.bid_increase

        elif flight.auction_on and len(flight.received_bids) == 1:
            flight.turn_count += 1
            winner = list(flight.received_bids[0].values())

        if winner:
            if (not flight.agents_in_my_formation) and (not winner[0].agents_in_my_formation):
                flight.start_formation(winner[0], winner[1], discard_received_bids=True)
            elif flight.agents_in_my_formation and flight.formation_role == "master" and (
                    not winner[0].agents_in_my_formation):
                flight.add_to_formation(winner[0], winner[1], discard_received_bids=True)
            elif flight.agents_in_my_formation and flight.formation_role == "master" and winner[
                0].agents_in_my_formation and winner[0].formation_role == "master":
                flight.add_to_formation(winner[0], winner[1], discard_received_bids=True)
            flight.highest_bid = False
            flight.turn_count = 0


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
                score = (fuel_saved*(1+alliance_status))/delay_time
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