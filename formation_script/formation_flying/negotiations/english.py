'''
# =============================================================================
# This file contains the function to do an English auction. 
# =============================================================================
'''
def do_english(flight):
    delta_T = 5  # steps after which a bid expires
    target_agent = flight
    reservation_price = :
    #utility function of a manager

    def utility_manager(flight):
        bids = flight.received_bids
        utilities = []
        for agent in bids:
            utility = 40 * agent.alliance + 0.3 * agent.calculate_potential_fuelsavings(flight) # TODO Delay time addition:
            utilities.append({"agent": agent, "utility": utility})
         sorted(utilities, key=lambda i:i["utility"])
    #Manager opens an english auction
    if flight.formation_state == 2 and flight.manager:
        flight.auctioneer = True







    #Behaviour of a contractor finding an auctioneer
    if flight.auctioneer and flight.formation_state == 0:
        targets = find flight.
        end

    #behaviour of a




# def do_English(flight):

'''
# =============================================================================
# This file contains the function to do a Japanese auction. 
# =============================================================================
'''


def do_Japanese(flight):
    # Auctioneer
    if flight.auctioneer and flight.accepting_bids:
        bids = flight.received_bids
        if not flight.highest_bid:
            higher_bids = []
            for bid in bids:
                # unpack bid
                # some utlity function: output some score
                if bid > flight.highest_bid:
                    higher_bids.append(bid)
            # Order higher_bids per SCORE
            # Select highest score to become highest bid (save new value for flight.highest_bid)

    # Bidder
    if not flight.auctioneer and flight.formation_state == 0:
        candidates = flight.find_auction_candidates()
        scores = []
        bid_ratio = 0.4
        if candidates:
            for target_agent in candidates:
                alliance_status = target_agent.alliance
                delay_time = flight.calculate_delay(target_agent)
                fuel_saved = flight.calculate_potential_fuelsavings(target_agent)
                bid = bid_ratio * fuel_saved
                score = 500 * alliance_status - delay_time * 10 + fuel_saved
                scores.append({"agent": target_agent, "score": score, "bid": bid})
            sorted_scores = sorted(scores, key=lambda i: i["score"], reverse=True)
            for option in range(len(sorted_scores)):
                current_option = sorted_scores[option]
                if current_option[2] > current_option[0].highest_bid:
                    best_option = sorted_scores[option]
            # TODO Add make_bid alliance status flight.make_bid(best_option[0], best_option)