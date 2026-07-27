from unittest.mock import Mock

import numpy as np

from BSCSimulator.metrics import IntermediateMetrics, siloed_future_demand_supply_constraint, transit_constraint


def test_transit_constraint_basic():
    # 3 supply, 2 demand, 7 days, transit times are all 1
    transit_time_data = np.ones((3, 2, 7))
    supply_ids = np.array([1, 2, 0, 2, 1])
    demand_ids = np.array([0, 1])
    unit_available_dates = np.array([1, 2, 3, 1, 12])
    reqs_dates = np.array([2, 4])
    penalty_value = 1000

    penalty = transit_constraint(transit_time_data, supply_ids,
                                 demand_ids, unit_available_dates, reqs_dates, penalty_value)
    # All units should be able to arrive on time except if arrival_dates > reqs_dates
    # For this setup, arrival_dates = unit_available_dates + 1
    expected = np.zeros((2, 5))
    # e.g., unit 5 available at day 12, arrives at 13, request at 2 (too late); unit 1 available at day 1, arrives at 2, request at 2 (ok)
    expected[0, [False, True, True, False, True]] = penalty_value
    # e.g., unit 3 available at day 3, arrives at 4, request at 4 (ok); unit 2 available at day 2, arrives at 3, request at 4 (ok)
    expected[1, [False, False, False, False, True]] = penalty_value
    assert penalty.shape == expected.shape
    assert np.all((penalty == 0) | (penalty == penalty_value))


def test_transit_constraint_penalty_applied():
    # Setup where all arrivals are too late
    transit_time_data = np.full((3, 2, 7), 5)
    supply_ids = np.array([2, 1, 0, 2, 1, 0])
    demand_ids = np.array([1, 0, 1])
    unit_available_dates = np.array([1, 1, 1, 1, 1, 1])
    reqs_dates = np.array([2, 2, 2])
    penalty_value = 9999

    penalty = transit_constraint(transit_time_data, supply_ids,
                                 demand_ids, unit_available_dates, reqs_dates, penalty_value)
    assert np.all(penalty == penalty_value)


def test_transit_constraint_zero_transit_time():
    # Setup where transit time is zero, all units should be usable
    transit_time_data = np.zeros((2197, 4913, 7))
    supply_ids = np.random.randint(0, 2197, size=100)
    demand_ids = np.random.randint(0, 4913, size=10)
    unit_available_dates = np.random.randint(1, 10, size=100)
    reqs_dates = np.random.randint(9, 15, size=10)
    penalty_value = 1234

    penalty = transit_constraint(transit_time_data, supply_ids,
                                 demand_ids, unit_available_dates, reqs_dates, penalty_value)
    assert np.all(penalty == 0)
    # Setup where transit time is zero, but some units are available after request dates
    # Make first 10 units available after all request dates
    unit_available_dates[:10] = 20
    penalty = transit_constraint(transit_time_data, supply_ids,
                                 demand_ids, unit_available_dates, reqs_dates, penalty_value)
    assert np.all(penalty[:, :10] == penalty_value)
    assert np.all(penalty[:, 10:] == 0)


def test_transit_constraint_edge_case_empty():
    # Edge case: no supply or demand
    transit_time_data = np.ones((0, 0, 7), dtype=int)
    supply_ids = np.array([], dtype=int)
    demand_ids = np.array([], dtype=int)
    unit_available_dates = np.array([], dtype=int)
    reqs_dates = np.array([], dtype=int)
    penalty_value = 1e16

    penalty = transit_constraint(transit_time_data, supply_ids,
                                 demand_ids, unit_available_dates, reqs_dates, penalty_value)
    assert penalty.size == 0


def test_transit_constraint_select_correct_transit_times():
    # Setup with varying transit times
    transit_time_data = np.array([[[3, 1, 2, 3, 2, 1, 4],    # Supply 0 to Demand 0 over 7 days
                                   [3, 4, 0, 3, 2, 3, 0]],   # Supply 0 to Demand 1 over 7 days
                                  [[0, 1, 0, 3, 4, 4, 0],    # Supply 1 to Demand 0 over 7 days
                                   [4, 3, 3, 1, 2, 2, 3]],   # Supply 1 to Demand 1 over 7 days
                                  [[4, 2, 3, 0, 0, 4, 3],    # Supply 2 to Demand 0 over 7 days
                                   [1, 3, 0, 4, 3, 0, 2]]])  # Supply 2 to Demand 1 over 7 days
    # Shape (3 supply locations, 2 demand locations, 7 days)
    supply_ids = np.array([0, 1, 2, 0, 1, 2])
    demand_ids = np.array([1, 0, 1])
    unit_available_dates = np.array([1, 1, 3, 7, 12, 13])
    reqs_dates = np.array([3, 8, 20])
    # Therefore, the maximum transit times to meet requests are below:
    # maximum_transit_times = np.array([[  2,   2,   0,  -4,  -9, -10],
    #                                   [  7,   7,   5,   1,  -4,  -5],
    #                                   [ 19,  19,  17,  13,   8,   7]])
    # Correct transit times selected should be:
    # transit_times_selected = np.array([[3, 4, 0, 0, 2, 0],    # Demand 0
    #                                    [3, 0, 3, 4, 4, 4],    # Demand 1
    #                                    [3, 4, 0, 0, 2, 0]])   # Demand 2
    penalty_value = 500
    penalty = transit_constraint(transit_time_data, supply_ids,
                                 demand_ids, unit_available_dates, reqs_dates, penalty_value)
    expected = np.array([[penalty_value, penalty_value, 0, penalty_value, penalty_value, penalty_value], # Demand 0
                         [0, 0, 0, penalty_value, penalty_value, penalty_value],                         # Demand 1
                         [0, 0, 0, 0, 0, 0]])                                                            # Demand 2
    assert penalty.shape == expected.shape
    assert np.all((penalty == 0) | (penalty == penalty_value))
    assert np.array_equal(penalty, expected)
