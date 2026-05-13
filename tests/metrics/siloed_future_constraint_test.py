from unittest.mock import Mock

import numpy as np

from BSCSimulator.metrics import IntermediateMetrics, siloed_future_demand_supply_constraint


def test_siloed_future_demand_supply_constraint():
    num_today_requests = 2
    num_future_requests = 1
    stepwise_results = Mock(spec=IntermediateMetrics,)
    stepwise_results.transit_times = np.array([[0, 1, 1, 0, 1, 1, 0, 1],
                                               [1, 1, 0, 1, 0, 0, 1, 1],
                                               [0, 0, 1, 1, 0, 0, 0, 1],
                                               [1, 1, 0, 1, 1, 1, 1, 0],
                                               [0, 0, 0, 1, 1, 1, 0, 0]])
    penalty_value = 1e16
    penalty = siloed_future_demand_supply_constraint(num_today_requests, num_future_requests,
                                                     stepwise_results, penalty_value)
    expected = np.zeros((5, 8))
    expected[2:3, :] = [0, 0, penalty_value, penalty_value, 0, 0, 0, penalty_value]
    assert penalty.shape == expected.shape
    assert np.array_equal(penalty, expected)
    assert np.all((penalty == 0) | (penalty == penalty_value))
    
