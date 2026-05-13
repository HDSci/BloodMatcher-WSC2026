import datetime
import logging
import os
import time

import numpy as np
import pandas as pd

from BSCSimulator.antigen import Antigens
from BSCSimulator.bloodgroups import ANTIGENS
from BSCSimulator.data import DataIO
from BSCSimulator.location import Locations
from BSCSimulator.location.inventory import LocationInventory
from BSCSimulator.matching import MatchingArea
from BSCSimulator.simulator import SimulationManager

logger = logging.getLogger(__name__)


def precompute_exp4(anticipation=False, seed=0xBE_BAD_BAE, cpus=1, replications=200, folder=None, **kwargs):
    """Pre-computation for Experiment 4

    :param bool anticipation: Whether to use anticipation. Redundant for this function.
    :param int seed: The seed for the random number generator(s).
    :param int cpus: The number of CPUs to parallelise precomputation 'replications'.
    :param int replications: The number of simulation replications precomputation is required for.
    :param str folder: The folder to save the pre-computed data.
    """
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime('%Y%m%d-%H-%M')
    print(f'\n###\nPre-computing simulations at {start_datetime}:')

    folder = folder if folder is not None else os.path.realpath(f'out/experiments/precompute/{start_datetime}')
    folder = os.path.realpath(folder)
    
    bsc_data = DataIO(folder)
    allo_ab_data = bsc_data.load_alloantibodies(kwargs.get('ab_datafile'))
    bsc_data.load_supply(kwargs.get('supply_data'))
    bsc_data.load_demand(kwargs.get('demand_data'))
    bsc_data.load_manufacturing_pathways(kwargs.get('manufacturing_pathways_data'))
    immuno = bsc_data.load_immunogenicity(kwargs.get('immunogenicity_data'))
    bsc_data.load_transportation_data(kwargs.get('courier_costs_data'), kwargs.get('transit_time_data'))
    mismatch = bsc_data.load_mismatch_parameters(kwargs.get('mismatch_data'))
        
    os.makedirs(folder, exist_ok=True)
    
    # unpack warm_up, horizon, cool_down from kwargs if present else use defaults   
    warm_up = kwargs.get('warm_up', 7 * 6 * 4)  # 4 weeks
    horizon = kwargs.get('horizon', 7 * 6 * 5)  # 5 weeks
    cool_down = kwargs.get('cool_down', 0)  # 0 weeks

    _anticipation = anticipation #and rule == 'Extended'

    antigens = Antigens(ANTIGENS,
                        alloimmunisation_risk=immuno,
                        mismatch_weights=mismatch,
                        allo_Abs=allo_ab_data.values.flatten(),)

    def locations(loc_rngs, loc_seeds):
        return Locations(bsc_data.location_data, bsc_data.donor_data,
                            bsc_data.patient_data, antigens, loc_rngs, loc_seeds)
    def demand(): return None
    def supply(): return None
    def matching():
        return MatchingArea(
            algo='transport', antigens=antigens, anticipation=_anticipation,
            cost_weights=None, solver=kwargs.get('solver', 'maxflow'),
            young_blood_constraint=kwargs.get('yb_constraint', True),
            substitution_penalty_parity=kwargs.get('substitution_weight_equal', True),
            unit_moving_window=kwargs.get('unit_moving_window', 0))
    def inventory():
        return LocationInventory(kwargs.get('max_age', 35),
            inventory_size=kwargs.get('inventory_size', 30_000),
            watched_antigens=kwargs.get('watched_antigens', np.array([114688])),
            watched_phenotypes=kwargs.get('watched_phenotypes', np.array([0])),
            watched_phenotype_names=kwargs.get('watched_phenotypes_names', ['O-']),)
    manager = SimulationManager(antigens, demand, supply, matching, inventory,
                                warm_up, horizon, cool_down, replications, seed,
                                locations=locations, precompute_outfolder=folder,)
    manager.do_precompute(cpus)

    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60

    print(f'\nThe precompute took {elapsed_mins: .1f} minute(s).')
    print(f'\nThe output folder is at {folder}')


def tuning(seed=0xBE_BAD_BAE, replications=10, cpus=10, weights: np.ndarray = None,
           num_objectives=1, anticipation=True, name='TuningSim', **kwargs):
    """Tuning of the simulation parameters
    
    :param int seed: The seed to use, defaults to 0xBE_BAD_BAE
    :param int cpus: The number of CPUs to use, defaults to 10
    :param int replications: The number of simulation replications to run, defaults to 10
    :param np.ndarray weights: The penalty weights for the cost function, defaults to None
    :param int num_objectives: The number of objectives to return, defaults to 1
    :param bool anticipation: Whether to use anticipation, defaults to True
    :param str name: The name for the output files, defaults to 'TuningSim'
    :param kwargs: Additional keyword arguments for the simulation, such as data file paths and simulation parameters.
    :return:
    """
    start_datetime = datetime.datetime.now().strftime('%Y%m%d-%H-%M')
    root_now = datetime.datetime.now()
    root_folder_date = root_now.strftime('%Y%m%d')
    root_folder_time = root_now.strftime('%H%M')
    exp: str = '4' if kwargs.get('exp', None) is None else kwargs.get('exp', '4')
    folder = kwargs.get('folder', os.path.join(
        f'out/experiments/exp{exp}/tuning', root_folder_date, root_folder_time, ''))    
    
    print(f'\n###\nTuning evaluation at {start_datetime}')
    bsc_data = DataIO(folder, tuning=True)
    allo_ab_data = bsc_data.load_alloantibodies(kwargs.get('ab_datafile'))
    bsc_data.load_supply(kwargs.get('supply_data'))
    bsc_data.load_demand(kwargs.get('demand_data'))
    # bsc_data.load_routing(kwargs.get('routing_data'))
    bsc_data.load_manufacturing_pathways(kwargs.get('manufacturing_pathways_data'))
    immuno = bsc_data.load_immunogenicity(kwargs.get('immunogenicity_data'))
    bsc_data.load_transportation_data(kwargs.get('courier_costs_data'), kwargs.get('transit_time_data'))
    mismatch = bsc_data.load_mismatch_parameters(kwargs.get('mismatch_data'))
    pre_compute_folder = os.path.realpath(kwargs.get('pre_compute_folder'))

    # unpack warm_up, horizon, cool_down from kwargs if present else use defaults
    warm_up = kwargs.get('warm_up', 7 * 6 * 4)  # 4 weeks
    horizon = kwargs.get('horizon', 7 * 6 * 5)  # 5 weeks
    cool_down = kwargs.get('cool_down', 0)  # 0 weeks

    forecasting = kwargs.get('forecasting', None)
    rebalance_stock = kwargs.get('rebalance_stock', False)
    rebalance_stock_days = kwargs.get('rebalance_stock_days', None)
    rebalance_lead_time = kwargs.get('rebalance_lead_time', 1)

    if isinstance(anticipation, bool):
        # _anticipation = [anticipation and rule == 'Extended'] * 3
        _anticipation = [anticipation] * 3
    else:
        # _anticipation = [antn and rule == 'Extended' for antn in anticipation]
        _anticipation = anticipation

    antigens = Antigens(ANTIGENS,
                        alloimmunisation_risk=immuno,
                        mismatch_weights=mismatch,
                        allo_Abs=allo_ab_data.values.flatten(),)
    def locations(loc_rngs, loc_seeds):
        return Locations(bsc_data.location_data, bsc_data.donor_data,
                         bsc_data.patient_data, antigens, loc_rngs, loc_seeds,
                         bsc_data.courier_data_df, bsc_data.transit_time_data_dfs,)
    def demand(): return None
    def supply(): return None
    def matching():
        return MatchingArea(
            algo='transport', antigens=antigens, anticipation=_anticipation[0],
            cost_weights=weights, solver=kwargs.get('solver', 'maxflow'),
            young_blood_constraint=kwargs.get('yb_constraint', True),
            substitution_penalty_parity=kwargs.get('substitution_weight_equal', True),
            unit_moving_window=kwargs.get('unit_moving_window', 0),
            discount_future=kwargs.get('discount_future_demand'),
            expert_mismatch_risk=kwargs.get('expert_mismatch_risk', False),
            multi_stage_matching=kwargs.get('multi_stage_matching'),
            multi_stage_strictness_factors=kwargs.get('multi_stage_matching_strictness'),
            random_relaxation=kwargs.get('random_relaxation', False),
            siloed_future=kwargs.get('siloed_shus', False),
            mismatching_parameters=bsc_data.mismatch_parameters,
            exact_matching_patient_groups=kwargs.get('exact_matching_patient_groups', []),
            mismatch_normalisation=kwargs.get('mismatch_function_normalisation', True),)
    # TODO: Make sure `watched_phenotypes` starts with O- and O+
    def inventory():
        return LocationInventory(
            kwargs.get('max_age', 35),
            inventory_size=kwargs.get('inventory_size', 30_000),
            watched_antigens=kwargs.get('watched_antigens', np.array([114688])),
            watched_phenotypes=kwargs.get('watched_phenotypes', np.array([0])),
            watched_phenotype_names=kwargs.get('watched_phenotypes_names', ['O-']),)
    pre_compute_folder = pre_compute_folder if _anticipation[1] or _anticipation[2] else None
    manager = SimulationManager(antigens, demand, supply, matching, inventory,
                                warm_up, horizon, cool_down, replications, seed,
                                pre_compute_folder,  _anticipation[1],
                                _anticipation[2], forecasting=forecasting,
                                locations=locations, rebalance_stock=rebalance_stock,
                                rebalance_stock_days=rebalance_stock_days,
                                rebalance_lead_time=rebalance_lead_time,)
    if cpus > 1 and replications > 1:
        manager.do_simulations_parallel(min(cpus, replications))
    else:
        manager.do_simulations()
    manager.statistics()
    
    objectives_values = bsc_data.save_output(manager, ANTIGENS, name,
                                             kwargs.get('watched_phenotypes_names', ['O-']),
                                             kwargs.get('computation_times', False),
                                             num_objectives,
                                             kwargs.get('objectives_names'))
    if num_objectives == 1:
        return objectives_values[0]
    else:
        return objectives_values


def exp4(anticipation=False, seed=0xBE_BAD_BAE, cpus=1, replications=60, weights=None, name='Simulation',
         **kwargs):
    """Experiment 4 - Simulation with locations for supply and demand.

    :param bool anticipation: Whether to use anticipation.
    :param int seed: The seed for the random number generator(s).
    :param int cpus: The number of CPUs to parallelise simulation replications.
    :param dict weights: The penalty weights for the cost function.
    :param int replications: The number of simulation replications to run.
    :param str name: The name for the output files, defaults to 'Simulation'.
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime('%Y%m%d-%H-%M')
    exp: str = '4' if kwargs.get('exp', None) is None else kwargs.get('exp', '4')

    folder = _handle_slurm_folder_clash(start_time, exp)

    print(f'\n###\nStarting simulations at {start_datetime}:')
    bsc_data = DataIO(folder)
    allo_ab_data = bsc_data.load_alloantibodies(kwargs.get('ab_datafile'))
    bsc_data.load_supply(kwargs.get('supply_data'))
    bsc_data.load_demand(kwargs.get('demand_data'))
    # bsc_data.load_routing(kwargs.get('routing_data'))
    bsc_data.load_manufacturing_pathways(kwargs.get('manufacturing_pathways_data'))
    immuno = bsc_data.load_immunogenicity(kwargs.get('immunogenicity_data'))
    bsc_data.load_transportation_data(kwargs.get('courier_costs_data'), kwargs.get('transit_time_data'))
    mismatch = bsc_data.load_mismatch_parameters(kwargs.get('mismatch_data'))
    pre_compute_folder = os.path.realpath(kwargs.get('pre_compute_folder'))

    # unpack warm_up, horizon, cool_down from kwargs if present else use defaults
    warm_up = kwargs.get('warm_up', 7 * 6 * 4)  # 4 weeks
    horizon = kwargs.get('horizon', 7 * 6 * 5)  # 5 weeks
    cool_down = kwargs.get('cool_down', 0)  # 0 weeks

    forecasting = kwargs.get('forecasting', None)
    rebalance_stock = kwargs.get('rebalance_stock', False)
    rebalance_stock_days = kwargs.get('rebalance_stock_days', None)
    rebalance_lead_time = kwargs.get('rebalance_lead_time', 1)

    if isinstance(anticipation, bool):
        _anticipation = [anticipation] * 3
    else:
        _anticipation = anticipation

    antigens = Antigens(ANTIGENS,
                        alloimmunisation_risk=immuno,
                        mismatch_weights=mismatch,
                        allo_Abs=allo_ab_data.values.flatten(),)

    def locations(loc_rngs, loc_seeds):
        return Locations(bsc_data.location_data, bsc_data.donor_data,
                            bsc_data.patient_data, antigens, loc_rngs, loc_seeds,
                            bsc_data.courier_data_df, bsc_data.transit_time_data_dfs,)
    def demand():
        return None
    def supply():
        return None
    def matching():
        return MatchingArea(
            algo='transport', antigens=antigens,
            anticipation=_anticipation[0],
            cost_weights=weights, solver=kwargs.get('solver', 'maxflow'),
            young_blood_constraint=kwargs.get('yb_constraint', True),
            substitution_penalty_parity=kwargs.get('substitution_weight_equal', True),
            unit_moving_window=kwargs.get('unit_moving_window', 0),
            penalty_names=kwargs.get('penalty_names', None),
            discount_future=kwargs.get('discount_future_demand'),
            expert_mismatch_risk=kwargs.get('expert_mismatch_risk', False),
            multi_stage_matching=kwargs.get('multi_stage_matching'),
            multi_stage_strictness_factors=kwargs.get('multi_stage_matching_strictness'),
            random_relaxation=kwargs.get('random_relaxation', False),
            siloed_future=kwargs.get('siloed_shus', False),
            mismatching_parameters=bsc_data.mismatch_parameters,
            exact_matching_patient_groups=kwargs.get('exact_matching_patient_groups', []),
            mismatch_normalisation=kwargs.get('mismatch_function_normalisation', True),)
    # TODO: Make sure `watched_phenotypes` starts with O- and O+
    def inventory():
        return LocationInventory(
            kwargs.get('max_age', 35),
            inventory_size=kwargs.get('inventory_size', 30_000),
            watched_antigens=kwargs.get(
                'watched_antigens', np.array([114688])),
            watched_phenotypes=kwargs.get(
                'watched_phenotypes', np.array([0])),
            watched_phenotype_names=kwargs.get(
                'watched_phenotypes_names', ['O-']),)
    pre_compute_folder = pre_compute_folder if _anticipation[1] or _anticipation[2] else None
    manager = SimulationManager(
        antigens, demand, supply, matching, inventory, warm_up, horizon, cool_down,
        replications, seed,
        pre_compute_folder, _anticipation[1], _anticipation[2], forecasting=forecasting,
        locations=locations, rebalance_stock=rebalance_stock, rebalance_stock_days=rebalance_stock_days,
        rebalance_lead_time=rebalance_lead_time,)
    if cpus > 1 and replications > 1:
        manager.do_simulations_parallel(min(cpus, replications))
    else:
        manager.do_simulations()
    manager.statistics()

    bsc_data.save_output(manager, ANTIGENS, name,
                            kwargs.get('watched_phenotypes_names', ['O-']),
                            kwargs.get('computation_times', False), 0, [])

    end_time = time.time()
    elapsed_mins = (end_time - start_time) / 60

    print(f'\nThe elapsed time so far is {elapsed_mins: .1f} minute(s).')

    print(f'\nThe output folder is at {folder}')


def _handle_slurm_folder_clash(start_time: float, exp: str) -> str:
    """Handle and resolve folder clash when running on a SLURM cluster."""

    folder_clash = True
    folder_clash_count = 0
    folder_clash_max = 250
    folder_clash_rng = np.random.default_rng(int(start_time * 1000))
    while folder_clash and folder_clash_count < folder_clash_max:
        root_now = datetime.datetime.now()
        root_folder_date = root_now.strftime('%Y%m%d')
        root_folder_time = root_now.strftime('%H%M')

        folder = os.path.join(
            f'out/experiments/exp{exp}', root_folder_date, root_folder_time, '')
        folder = os.path.realpath(folder)
        if os.path.exists(folder):
            folder_clash_count += 1
            time.sleep(20 * folder_clash_rng.random())
        else:
            try:
                os.makedirs(folder, exist_ok=False)
                folder_clash = False
            except OSError:
                folder_clash_count += 1
                time.sleep(20 * folder_clash_rng.random())

    if folder_clash_count == folder_clash_max:
        raise OSError(
            f'Could not create unique output folder after {folder_clash_max} attempts.')

    return folder
