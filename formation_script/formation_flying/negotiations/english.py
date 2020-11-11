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