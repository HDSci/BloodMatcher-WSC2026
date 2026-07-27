import numpy as np


def test_precompute_same_as_supply_demand():
    last_day_supply_demand_file = 'scratch/manual_tests/forecasting/last_day_forecast.npz'
    precomputed_file = 'out/experiments/precompute/exp4/supply_chain/2025-old-dummy-demand/00000_0XBEBADBAE_randvars.npz'
    
    last_day_supply_demand = np.load(last_day_supply_demand_file)
    precomputed = np.load(precomputed_file)

    # last_unit_id = last_day_supply_demand['donated_units'][-1, 0]
    last_unit_id = last_day_supply_demand['routed_available_units'][-1, 0]
    last_demand_id = last_day_supply_demand['new_requests'][-9, 0]
    
    assert np.allclose(precomputed['units'][precomputed['units'][:,0] == last_unit_id],
                last_day_supply_demand['routed_available_units'][-1, :])
    assert np.allclose(precomputed['requests'][precomputed['requests'][:,0] == last_demand_id],
                last_day_supply_demand['new_requests'][-9, :])
    assert np.allclose(precomputed['requests_Abs'][precomputed['requests'][:,0] == last_demand_id],
                last_day_supply_demand['abs_mask'][-9, :])