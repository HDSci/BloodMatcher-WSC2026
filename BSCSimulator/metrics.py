import numexpr as ne
import numpy as np


class IntermediateMetrics:
    """Class to hold intermediate metrics/ stepwise results for later use, avoiding recalculation."""

    def __init__(self):
        self.transit_times: np.ndarray = None
        self.units_arriving_exactly_on_time: np.ndarray = None
        self.units_arriving_exactly_on_time_with_buffer: np.ndarray = None
        self.early_arrival_buffer: float = 0
        self.allocating_units_to_the_past: np.ndarray = None
        self.units_will_expire_before_request: np.ndarray = None
        self.young_blood_will_be_old_blood: np.ndarray = None


def phenotype_usability(supply, demand):
    assert len(demand.shape) == 2
    usability_norm = len(demand)
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    usability = np.all(demand_rotated >= supply, axis=2).sum(axis=0) / usability_norm
    return usability


def pop_phenotype_usability(abd_phenotypes, frequencies):
    usability = frequencies[abd_phenotypes.flatten()]
    return usability


def sum_of_substitutions(supply, demand, weights=None):
    assert len(demand.shape) == 2
    weights = np.ones(demand.shape[1]) if weights is None else weights
    substitution_norm = weights.sum()
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    substitutions = ne.evaluate('sum((demand_rotated > supply) * weights, axis=2)') / substitution_norm
    return substitutions


def mismatching(supply, demand, risk, normalise = True):
    assert len(demand.shape) == 2
    if normalise:
        risk_norm = risk.sum()
        risk_norm = 1 if risk_norm == 0 else risk_norm
    else:
        risk_norm = 1
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    immunogenicity = ne.evaluate('sum((demand_rotated < supply) * risk, axis=2)') / risk_norm
    return immunogenicity


def major_antigen_substitution(supply, demand, population=False, pop_frequencies=None):
    if population:
        Fsi = pop_phenotype_usability(supply, pop_frequencies)
        Fsj = pop_phenotype_usability(demand, pop_frequencies)
    else:
        Fsi = phenotype_usability(supply, demand)
        Fsj = phenotype_usability(demand, demand)
    Fsj = Fsj[:, None]
    mas = Fsi - Fsj
    return mas


# TODO: Make calculation of age constraints separate from the core FIFO penalty
# I.e., calculating and constraining forecasted units so that they cannot be used
# earlier than they are produced should be done in a separate function.
def fifo_discount(remaining_shelf_life, max_life=35, reqs_dates=None, scd_pat=False):
    if reqs_dates is None:
        discount = _fifo_discount(remaining_shelf_life)
        # Penalise forecasted units so they cannot be used for today's requests
        discount[remaining_shelf_life > max_life] = -1e16
        discount = discount[None, :]
        if np.any(scd_pat):
            len_demand = len(scd_pat)
            discount = np.full((len_demand, discount.size), discount)
            discount[scd_pat] = 0
    else:
        _remaining_shelf_life = remaining_shelf_life[None, :] - reqs_dates[:, None]
        discount = _fifo_discount(_remaining_shelf_life)
        allocating_to_the_past = (remaining_shelf_life - (max_life))[None, :] > reqs_dates[:, None]
        will_expire = _remaining_shelf_life < 0
        discount[allocating_to_the_past] = -1e16
        discount[will_expire] = -1e16
        if np.any(scd_pat):
            discount[scd_pat] = 0
    return discount


def young_blood_penalty(unit_age, max_young_blood=14, reqs_dates=None, scd_pat=None):
    """Young blood/old blood penalties and constraints.
    
    Adds a constraint so that blood that is not 'young blood' cannot be used
    for SCD patients.
    Sets penalties and bonuses so that older 'young blood' is used first
    up 7 days old, then between 7 and 14 days old, the penalties exponentially increase.
    
    :param unit_age: age minus 1 of the unit in days.
    :param max_young_blood: maximum age of young blood in days.
    :param reqs_dates: dates of the requests.
    :param scd_pat: boolean array indicating which requests are for SCD patients.
    :return: penalty array.    
    """
    if scd_pat is None:
        scd_pat = np.array([True], dtype=bool)
    if reqs_dates is None:
        penalty = _young_blood_penalty(unit_age + 1)
        # Penalise non-young blood so it cannot be used
        penalty[unit_age + 1 > max_young_blood] = 1e16
        penalty = penalty[None, :]
        if not np.all(scd_pat):
            len_demand = len(scd_pat)
            penalty = np.full((len_demand, penalty.size), penalty)
            penalty[~scd_pat] = 0
    else:
        _unit_age = unit_age[None, :] + reqs_dates[:, None]
        penalty = _young_blood_penalty(_unit_age + 1)
        allocating_to_the_past = _unit_age < 0
        penalty[allocating_to_the_past] = 1e16
        will_be_old_blood = _unit_age + 1 > max_young_blood
        penalty[will_be_old_blood] = 1e16
        if not np.all(scd_pat):
            penalty[~scd_pat] = 0
    return penalty


def _fifo_discount(remaining_shelf_life: np.ndarray, decay_period:int=3,):
    """
    Calculate the FIFO discount based on the remaining shelf life of the units.
    
    The discount is calculated as `discount = 0.5 ** (remaining_shelf_life / decay_period)`.
    This means that the discount is 1 for units at the end of their shelf life,
    and decreases exponentially as the remaining shelf life increases, with a default decay period of 3 days.
    
    :param ndarray remaining_shelf_life: 1-D array of remaining shelf life of the units in days
    :param int decay_period: decay period for the FIFO discount in days
    :return: 1-D ndarray of FIFO discounts for the units
    """
    max_abs = np.max(np.abs([np.max(remaining_shelf_life), np.min(remaining_shelf_life)]))
    poss_shelf_lifes = np.hstack((np.arange(max_abs + 1), np.arange(-max_abs, 0)))
    shelf_life_lookups = 0.5 ** (poss_shelf_lifes / decay_period)  # Changed to 3 because faster decay than 5 days for shorter 23/24 day shelf life(s)
    discount = shelf_life_lookups[remaining_shelf_life]
    return discount


def _young_blood_penalty(unit_age, a=7.686455, b=9.580724, c=0, d=1.1976):
    max_abs = np.max(np.abs(unit_age))
    poss_unit_ages = np.hstack((np.arange(max_abs + 1), np.arange(-max_abs, 0)))
    unit_age_lookups = (-1/a * poss_unit_ages + np.exp(poss_unit_ages - b) + c) * d
    penalty = unit_age_lookups[unit_age.astype(int)]
    return penalty


def can_supply_go_to_demand(supply_ids: np.ndarray, demand_ids: np.ndarray, penalty_value = 1e16):
    """Check if units at the supply points can be used to satisfy demand at the demand points.
    
    The check is done by comparing the ID of the location where each unit is with the
    demand ID of the the location where the request originates.
    They can be used if the modulo of the demand ID with the ID is zero.
    I.e., if the demand ID is a multiple of the ID.
    
    :param ndarray supply_ids: 1-D integer array of location IDs of the units
    :param ndarray demand_ids: 1-D integer array of demand IDs of the patient requests
    :param float penalty_value: penalty value for units that cannot be used
    :return ndarray: 2-D penalty array
    """
    demand_ids = demand_ids[:, None]
    supply_ids = supply_ids[None, :]
    location_connections = ne.evaluate('(demand_ids % supply_ids) == 0')
    penalty = ~location_connections * penalty_value    
    return penalty


# TODO: Need to maybe somehow check that JIT dispatching of units does not lead to a situation
# where sending a unit on day t, would mean it arrives too early, but sending it on day t+1 would mean it arrives after the request date 
def transit_constraint(transit_time_data: np.ndarray, supply_ids: np.ndarray, demand_ids: np.ndarray,
                       unit_available_dates: np.ndarray, reqs_dates: np.ndarray, penalty_value = 1e16,
                       stepwise_results: IntermediateMetrics=None) -> np.ndarray:
    """Check if units at the supply points can be used to satisfy demand at the demand points,
    taking into account the transit time between supply and demand points.
    
    The check is done by determining the transit time between each supply and demand point
    and comparing the date when the unit will be available at the supply point
    plus the transit time (which is conditional on the day of the week that the unit is available/is sent)
    with the date of the request.
    
    For a (future) unit its earliest `available_date` plus `transit_time` must be ≤ a future request's `due_date`.
    The `transit_time` needs to be calculated based on the day of the week `available_date` falls on.
    
    :param ndarray transit_time_data: 3-D array of transit times between supply locations for each day of the week
    :param ndarray supply_ids: 1-D integer array of location IDs of the units
    :param ndarray demand_ids: 1-D integer array of location sink IDs of the patient requests
    :param ndarray unit_available_dates: 1-D array of dates when the units will be available at the supply point
    :param ndarray reqs_dates: 1-D array of dates of the requests
    :param float penalty_value: penalty value for units that cannot be used
    :return ndarray: 2-D penalty array
    """
    d = len(reqs_dates)
    s = len(supply_ids)
    day_of_week = (unit_available_dates - 1) % 7
    transit_times = transit_time_data[np.repeat(supply_ids, d).reshape(s, d),
                                      np.tile(demand_ids, s).reshape(s, d),
                                      day_of_week[:, None]].T
    unit_available_dates = unit_available_dates[None, :].repeat(d, axis=0)
    reqs_dates = reqs_dates[:, None].repeat(s, axis=1)
    arrival_dates = unit_available_dates + transit_times
    can_arrive_on_time = arrival_dates <= reqs_dates
    penalty = ~can_arrive_on_time * penalty_value
    if stepwise_results is not None and isinstance(stepwise_results, IntermediateMetrics):
        stepwise_results.transit_times = transit_times
        stepwise_results.units_arriving_exactly_on_time = arrival_dates == reqs_dates
        stepwise_results.units_arriving_exactly_on_time_with_buffer = can_arrive_on_time & (arrival_dates >= reqs_dates - stepwise_results.early_arrival_buffer)
    return penalty


def siloed_future_demand_supply_constraint(num_today_requests, num_future_requests,
                                           stepwise_results: IntermediateMetrics, penalty_value=1e16):
    """Constraint to enforce no matching of units to demand across locations when the supply or demand or both are in the future.
    
    :param int num_today_requests: number of requests that are for today
    :param int num_future_requests: number of requests that are for the future
    :param IntermediateMetrics stepwise_results: store of working/intermediate calculations including retrieved transit times
    :param float penalty_value: penalty value for units that cannot be used
    :return ndarray: 2-D penalty array
    """
    penalties = np.zeros(stepwise_results.transit_times.shape, dtype=float)
    transit_above_zero = stepwise_results.transit_times[num_today_requests:(num_today_requests + num_future_requests), :] > 0
    penalties[num_today_requests:(num_today_requests + num_future_requests), :][transit_above_zero] = penalty_value
    return penalties


def can_unit_be_used_to_rebalance(supply_ids: np.ndarray, allowed_sources: list[np.ndarray],
                                  penalty_value = 1e16) -> np.ndarray:
    """Check if units at the supply points can be used to rebalance the inventory.
    
    The check is done by comparing the ID of the location where each unit is with the
    list of allowed sources for each demand point that needs to be rebalanced.
    
    :param ndarray supply_ids: 1-D integer array of location IDs of the units
    :param list[ndarray] allowed_sources: list of 1-D integer arrays of allowed source IDs for each demand point
    :param float penalty_value: penalty value for units that cannot be used
    :return ndarray: 2-D penalty array
    """
    penalties = np.full((len(allowed_sources), len(supply_ids)), penalty_value)
    for i, allowed_source in enumerate(allowed_sources):
        location_connections = np.isin(supply_ids, allowed_source)
        penalties[i, location_connections] = 0
    return penalties


def restrict_r0_units_from_rebalancing(supply_phenotypes: np.ndarray, watched_antigens: int = 31744, watched_phenotype: int = 21504, penalty_value = 1e16):
    """Restrict R0 units from being used to rebalance stock.
    
    The check is done by comparing the phenotype of the unit with the watched antigens and watched phenotype.
    If the unit is R0, it cannot be used to rebalance stock.
    
    :param ndarray supply_phenotypes: 1-D integer array of phenotypes of the units
    :param int watched_antigens: integer mask selecting the antigens in phenotype combination to watch for R0
    :param int watched_phenotype: integer phenotype combination to watch for R0
    :param float penalty_value: penalty value for units that cannot be used
    :return ndarray: 1-D penalty array
    """
    penalties = np.zeros(supply_phenotypes.shape, dtype=float)
    r0_units = (supply_phenotypes & watched_antigens) == watched_phenotype
    penalties[r0_units] = penalty_value
    return penalties[None, :]
    


def discount_future_demand(penalties: np.ndarray, demand_dates: np.ndarray, discount_strategy: str = 'flat',
                           rate: float = 0.95, init_step: float = 5, step: float = 0.2) -> np.ndarray:
    """Discount future demand based on the date of the request.
    
    In the 'flat' strategy, the discount is the same for all future days of demand:
    this is calculated as `new_penalty = old_penalty * rate + init_step`.
    
    In the 'variable' strategy, the discount increases by `step` for each subsequent day of demand:
    this is calculated as `new_penalty = (old_penalty * rate ** (day - date)) + init_step + step * (day - date - 1)`. 
    
    :param ndarray penalties: 2-D penalty array
    :param ndarray demand_dates: 1-D array of dates of the requests minus the current date
    :param str discount_strategy: 'flat' or 'variable'
    :param float rate:  1 minus discount rate
    :param int init_step: increase in penalty for the first future day of demand
    :param int step: increase in penalty for each subsequent day of demand
    :return ndarray: 2-D new penalty array
    """
    penalties = penalties.copy()
    if discount_strategy == 'flat':
        future_demand_indices = demand_dates > 0
        future_pens = penalties[future_demand_indices]
        penalties[future_demand_indices] = ne.evaluate('future_pens * rate + init_step')
    elif discount_strategy == 'variable':
        future_demand_indices = demand_dates > 0
        future_pens = penalties[future_demand_indices]
        dates_in_future = demand_dates[future_demand_indices][:, None]
        penalties[future_demand_indices] = ne.evaluate('(future_pens * (rate ** dates_in_future)) + init_step + (step * (dates_in_future - 1))')
    else:
        raise ValueError(f'Invalid discount strategy: {discount_strategy}.')
    return penalties


def transportation_cost(supply_ids: np.ndarray, demand_ids: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Calculate the transportation cost between supply and demand points.
    
    The cost is the normalised cost of a courier trip between the supply and demand points.
    
    :param ndarray supply_ids: 1-D integer array of location IDs of the units
    :param ndarray demand_ids: 1-D integer array of demand IDs of the patient requests
    :param ndarray cost: 2-D array of costs of a courier trip between the supply and demand points
    :return ndarray: 2-D cost array
    """
    normaliser = np.max(cost)
    normaliser = 1 if normaliser == 0 else normaliser
    normalised_cost = cost / normaliser
    d = len(demand_ids)
    s = len(supply_ids)
    transport_cost = np.zeros((d, s), dtype=float)
    transport_cost = normalised_cost[np.repeat(demand_ids, s).reshape(d, s),
                                     np.tile(supply_ids, d).reshape(d, s)]
    return transport_cost
