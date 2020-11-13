'''
# =============================================================================
# This file contains the function to do a Japanese auction. 
# =============================================================================
'''

def do_Japanese(flight):

    delta_T = 5

    # AUCTIONEER

    if flight.auctioneer and flight.accepting_bids:
        if flight.received_bids:
            if not flight.highest_bid:
                for bid in flight.received_bids:
                    current_bid = list(bid.values())
                    delay_time = flight.calculate_delay(current_bid[0])
                    score = 500 * current_bid[3] - delay_time * 30 + current_bid[1] * 2
                    bid["score"] = score
                sorted_bids = sorted(flight.received_bids, key=lambda i: i["score"])
                flight.highest_bid = sorted_bids[0]
                flight.turn_count = 0
                flight.accepting_bids = False
            if len(flight.received_bids) > 1:
                current_high = list(flight.highest_bid.values())
                new_desired = current_high[5]+abs(current_high[5]*0.05)
                flight.turn_count += 1
                for bid in flight.received_bids:
                    current_bid = list(bid.values())
                    if current_bid[5] < new_desired:
                        flight.received_bids.remove(bid)
                        current_bid[0].bid_made = False
                        current_bid[0].auctioneer_target = 0
                sorted_bids = sorted(flight.received_bids, key=lambda i: i["score"])
                flight.highest_bid = sorted_bids[0]
            if len(flight.received_bids) == 1:
                winner = list(flight.received_bids[0].values())
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
                bid = bid_ratio*fuel_saved
                score = 500*alliance_status-delay_time+fuel_saved
                scores.append({"agent": target_agent, "delay": delay_time, "alliance": alliance_status, "score": score, "bid": bid})
            sorted_scores = sorted(scores, key=lambda i: i["score"], reverse=True)
            best_option = list(sorted_scores[0].values())
            agent = best_option[0]
            delay_time = best_option[1]
            alliance_status = best_option[2]
            bid = best_option[4]
            flight.make_bid(agent, bid, delay_time, alliance_status, flight.model.schedule.steps+delta_T)
            flight.auctioneer_target = agent
            flight.bid_made = True