# Test Python parameter file for backward compatibility testing

main_func = 'BAYES_OPT'
seed = 999999
replications = 3
cpus = 2

simulation_time = {
    'warm_up': 50,
    'horizon': 100,
    'cool_down': 5
}

penalty_weights = {
    'immunogenicity': 2,
    'usability': 1,
    'substitutions': 1,
    'fifo': 0,
    'young_blood': 0,
    'transport_cost': 0
}

computation_times = True
