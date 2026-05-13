import datetime
import logging
import os
import sys

import multiprocess

from .experiments import (bayes_opt_tuning, exp4,
                          multi_objective_bayes_opt_tuning)
from .experiments import parameters as param
from .experiments import precompute_exp4


now = datetime.datetime.now().strftime('%Y%m%d')
log_folder = os.path.realpath('out/logs/')
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=f'{log_folder}/hpc_bsc_{now}.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    print(f'A total of {len(sys.argv)} arguments were passed.\n')
    if param.main_func == 'EXPERIMENT_4':
        exp4(seed=param.seed, replications=param.replications, cpus=param.cpus,
             anticipation=param.anticipation, weights=param.penalty_weights,
             penalty_names=param.penalty_names, exp=param.exp, name=param.name,
             **param.simulation_time, **param.stock_measurement,
             forecasting=param.forecasting, pre_compute_folder=param.pre_compute_folder,
             **param.data_files, solver=param.solver, **param.constraints,
             computation_times=param.computation_times, **param.stock_rebalancing,)
    elif param.main_func == 'PRECOMPUTE':
        precompute_exp4(seed=param.seed, cpus=param.cpus,
                        **param.simulation_time,
                        replications=param.replications, folder=param.pre_compute_folder,
                        **param.stock_measurement, **param.data_files,)
    elif param.main_func == 'BAYES_OPT':
        bayes_opt_tuning(**param.bayes_opt,
                         tuning_kwargs=dict(
                             cpus=param.cpus, seed=param.seed,
                             anticipation=param.anticipation,
                             **param.simulation_time,
                             **param.stock_measurement, forecasting=param.forecasting,
                             pre_compute_folder=param.pre_compute_folder,
                             **param.constraints, **param.data_files,
                             solver=param.solver, **param.stock_rebalancing,)
                         )
    elif param.main_func == 'MULTI_OBJECTIVE_BAYES_OPT':
        multi_objective_bayes_opt_tuning(**param.bayes_opt,
                                         tuning_kwargs=dict(
                                             cpus=param.cpus, seed=param.seed,
                                             anticipation=param.anticipation,
                                             **param.simulation_time,
                                             **param.stock_measurement,
                                             forecasting=param.forecasting,
                                             pre_compute_folder=param.pre_compute_folder,
                                             **param.constraints,
                                             **param.data_files, solver=param.solver,
                                             **param.stock_rebalancing,)
                                         )
    if len(sys.argv) > 2:
        print(*sys.argv[2:], sep='\n')
    print('\n------------------\n\n\n')
    return


if __name__ == "__main__":
    multiprocess.set_start_method('spawn', force=True)
    try:
        main()
        logger.info('Exiting normally.')
    except (Exception, KeyboardInterrupt):
        logger.exception('Exiting due to error.')
