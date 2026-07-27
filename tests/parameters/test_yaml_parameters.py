"""Tests for YAML parameter loading."""
import os
import tempfile
import numpy as np
import yaml


def test_yaml_parameter_loading():
    """Test that YAML parameters can be loaded correctly."""
    # Import the module to test default loading
    from BSCSimulator.experiments import parameters as param
    
    # Check that default parameters are loaded
    assert param.main_func == 'EXPERIMENT_4'
    assert param.seed == 0xBEBADBAE
    assert param.replications == 1
    assert param.cpus == 1
    assert param.solver == 'maxflow'
    
    # Check dictionary parameters
    assert isinstance(param.penalty_weights, dict)
    assert param.penalty_weights['immunogenicity'] == 1
    assert param.penalty_weights['fifo'] == 1
    
    # Check simulation time
    assert param.simulation_time['warm_up'] == 126
    assert param.simulation_time['horizon'] == 168
    
    # Check numpy arrays are created correctly
    assert isinstance(param.stock_measurement['watched_antigens'], np.ndarray)
    assert isinstance(param.stock_measurement['watched_phenotypes'], np.ndarray)
    assert len(param.stock_measurement['watched_antigens']) == 12
    assert len(param.stock_measurement['watched_phenotypes']) == 12


def test_yaml_file_structure():
    """Test that the default YAML file is well-formed."""
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..',
        'BSCSimulator',
        'experiments',
        'default_parameters.yml'
    )
    
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    # Check required top-level keys
    required_keys = [
        'main_func', 'seed', 'penalty_weights', 'simulation_time',
        'data_files', 'name', 'replications', 'cpus', 'stock_measurement',
        'stock_rebalancing', 'forecasting', 'bayes_opt', 'solver', 'constraints'
    ]
    
    for key in required_keys:
        assert key in params, f"Missing required key: {key}"
    
    # Validate data types
    assert isinstance(params['main_func'], str)
    assert isinstance(params['seed'], int)
    assert isinstance(params['penalty_weights'], dict)
    assert isinstance(params['simulation_time'], dict)
    assert isinstance(params['data_files'], dict)
    assert isinstance(params['name'], str)
    assert isinstance(params['replications'], int)
    assert isinstance(params['cpus'], int)


def test_custom_yaml_loading():
    """Test loading a custom YAML parameter file."""
    # Create a temporary YAML file
    custom_params = {
        'main_func': 'PRECOMPUTE',
        'seed': 99999,
        'replications': 5,
        'cpus': 4,
        'simulation_time': {
            'warm_up': 100,
            'horizon': 200,
            'cool_down': 10
        },
        'name': 'Test Experiment',
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(custom_params, f)
        temp_path = f.name
    
    try:
        # Import the load function
        from BSCSimulator.experiments.parameters import _load_yaml_parameters
        
        # Load the custom parameters
        loaded_params = _load_yaml_parameters(temp_path)
        
        # Verify loaded values
        assert loaded_params['main_func'] == 'PRECOMPUTE'
        assert loaded_params['seed'] == 99999
        assert loaded_params['replications'] == 5
        assert loaded_params['cpus'] == 4
        assert loaded_params['simulation_time']['warm_up'] == 100
        assert loaded_params['name'] == 'Test Experiment'
    finally:
        # Clean up
        os.unlink(temp_path)



