"""Integration test for YAML parameter loading."""
import os
import sys
import tempfile
import yaml


def test_yaml_parameter_integration():
    """Test the full integration of YAML parameter loading."""
    
    # Create a test YAML parameter file
    test_params = {
        'main_func': 'PRECOMPUTE',
        'seed': 54321,
        'replications': 2,
        'cpus': 1,
        'simulation_time': {
            'warm_up': 100,
            'horizon': 150,
            'cool_down': 10
        },
        'stock_measurement': {
            'inventory_size': 25000,
            'watched_antigens': [114688, 114688, 114688, 31744, 128, 64],
            'watched_phenotypes': [0, 16384, 32768, 49152, 65536, 81920]
        }
    }
    
    # Save to a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(test_params, f)
        temp_yaml_path = f.name
    
    # Save to a temporary Python file for backward compatibility test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("main_func = 'BAYES_OPT'\n")
        f.write("seed = 11111\n")
        f.write("replications = 5\n")
        temp_py_path = f.name
    
    try:
        # Test 1: Test YAML loading
        print("Test 1: YAML parameter loading")
        
        # Simulate command-line argument
        original_argv = sys.argv.copy()
        sys.argv = ['test', temp_yaml_path]
        
        # Import parameters module
        # Note: We need to reload to pick up the new argv
        import importlib
        from BSCSimulator.experiments import parameters
        importlib.reload(parameters)
        
        # Verify parameters were loaded
        assert parameters.main_func == 'PRECOMPUTE', f"Expected 'PRECOMPUTE', got {parameters.main_func}"
        assert parameters.seed == 54321, f"Expected 54321, got {parameters.seed}"
        assert parameters.replications == 2, f"Expected 2, got {parameters.replications}"
        print("  ✓ YAML parameters loaded correctly")
        
        # Test 2: Test Python file backward compatibility
        print("\nTest 2: Python parameter backward compatibility")
        
        sys.argv = ['test', temp_py_path]
        importlib.reload(parameters)
        
        assert parameters.main_func == 'BAYES_OPT', f"Expected 'BAYES_OPT', got {parameters.main_func}"
        assert parameters.seed == 11111, f"Expected 11111, got {parameters.seed}"
        assert parameters.replications == 5, f"Expected 5, got {parameters.replications}"
        print("  ✓ Python parameters loaded correctly (backward compatible)")
        
        # Test 3: Test default parameters (no argument)
        print("\nTest 3: Default parameter loading")
        
        sys.argv = ['test']
        importlib.reload(parameters)
        
        assert parameters.main_func == 'EXPERIMENT_4', f"Expected 'EXPERIMENT_4', got {parameters.main_func}"
        assert parameters.seed == 0xBEBADBAE, f"Expected 0xBEBADBAE, got {parameters.seed}"
        print("  ✓ Default parameters loaded correctly")
        
        # Restore original argv
        sys.argv = original_argv
        
        print("\n✅ All integration tests passed!")
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_yaml_path):
            os.unlink(temp_yaml_path)
        if os.path.exists(temp_py_path):
            os.unlink(temp_py_path)

