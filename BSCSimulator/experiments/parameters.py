"""
Parameter loading module for BSC Simulator.

This module provides configuration management for the Blood Supply Chain Simulator.
Parameters can be loaded from either YAML files (recommended) or Python files (legacy).

Default parameters are loaded from 'default_parameters.yml'. Custom parameters can be
provided via command-line argument to override defaults.

Usage:
    python -m BSCSimulator.main [parameterfile.yml]
    python -m BSCSimulator.main [parameterfile.py]  # Legacy support

The module exposes all parameters as module-level variables for easy access by
the main simulation code.
"""
import importlib.util
import logging
import os
import sys

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _load_yaml_parameters(yaml_path):
    """Load parameters from a YAML file."""
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    # Convert lists to numpy arrays for specific parameters
    if 'stock_measurement' in params:
        if 'watched_antigens' in params['stock_measurement']:
            params['stock_measurement']['watched_antigens'] = np.array(
                params['stock_measurement']['watched_antigens']
            )
        if 'watched_phenotypes' in params['stock_measurement']:
            params['stock_measurement']['watched_phenotypes'] = np.array(
                params['stock_measurement']['watched_phenotypes']
            )
    
    # Update bayes_opt replications to match top-level replications if not set
    if 'bayes_opt' in params and 'replications' in params:
        if params['bayes_opt'].get('replications') is None:
            params['bayes_opt']['replications'] = params['replications']
    
    return params


def _load_python_parameters(py_path):
    """Load parameters from a Python file (legacy support)."""
    pfile = os.path.split(py_path)[1]
    spec = importlib.util.spec_from_file_location(pfile[:-3], py_path)
    imported_param = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported_param)
    return {k: v for k, v in imported_param.__dict__.items() if not k.startswith('_')}


# Load default parameters from YAML
_default_params_path = os.path.join(
    os.path.dirname(__file__), 'default_parameters.yml'
)
_params = _load_yaml_parameters(_default_params_path)

# Set module-level variables from defaults
main_func = _params['main_func']
seed = _params['seed']
penalty_weights = _params['penalty_weights']
penalty_names = _params['penalty_names']
simulation_time = _params['simulation_time']
anticipation = _params['anticipation']
data_files = _params['data_files']
name = _params['name']
replications = _params['replications']
cpus = _params['cpus']
exp2 = _params['exp2']
exp = _params['exp']
stock_measurement = _params['stock_measurement']
stock_rebalancing = _params['stock_rebalancing']
forecasting = _params['forecasting']
pre_compute_folder = _params['pre_compute_folder']
bayes_opt = _params['bayes_opt']
solver = _params['solver']
constraints = _params['constraints']
computation_times = _params['computation_times']

# Load custom parameters if provided
if len(sys.argv) > 1:
    try:
        print(f'Parameter file passed: {sys.argv[1]}')
        pfile_path = os.path.realpath(os.path.expanduser(sys.argv[1]))
        
        # Determine file type and load accordingly
        if pfile_path.endswith('.yml') or pfile_path.endswith('.yaml'):
            print('Loading YAML parameter file...')
            custom_params = _load_yaml_parameters(pfile_path)
        elif pfile_path.endswith('.py'):
            print('Loading Python parameter file (legacy)...')
            custom_params = _load_python_parameters(pfile_path)
        else:
            raise ValueError(
                f'Unsupported parameter file format. Use .yml, .yaml, or .py'
            )
        
        # Update global namespace with custom parameters
        globals().update(custom_params)
        
    except IndexError as e:
        logger.exception('No parameter file passed. Using default parameters.')
        print('No parameter file passed. Using default parameters.')
    except FileNotFoundError as e:
        pfile = os.path.split(sys.argv[1])[1]
        logger.exception(f'Parameter file {pfile} not found. Using default parameters.')
        print(f'Parameter file {pfile} not found. Using default parameters.')
    except (ImportError, ValueError) as e:
        pfile = os.path.split(sys.argv[1])[1]
        logger.exception(f'Error loading parameter file {pfile}: {e}')
        print(f'Error loading parameter file {pfile}: {e}. Using default parameters.')
    except Exception as e:
        pfile = os.path.split(sys.argv[1])[1] if len(sys.argv) > 1 else 'unknown'
        logger.exception(f'Error while importing parameter file {pfile}. Using default parameters.')
        print(f'Error while importing parameter file {pfile}. Using default parameters.')
