# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from ortools.graph.python import min_cost_flow
import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.uint8_t uint8


def maxflow_mincost(cnp.ndarray supply, cnp.ndarray demand, cnp.ndarray pens, cnp.ndarray scaled_cost):
    cdef int n = supply.size
    cdef int m = demand.size
    cdef int tot = max(supply.sum(), demand.sum())
    # cdef cnp.ndarray pens = cost < 1e17
    # cdef int pen_sum = np.count_nonzero(pens)
    cdef int[:,:] true_penalty = (pens).astype(np.intc)
    scaled_cost = scaled_cost * 1000
    return _maxflow(supply.astype(np.intc), -demand.astype(np.intc), scaled_cost.astype(long), n, m, tot, true_penalty)


cdef cnp.ndarray _maxflow(int[:] supply, int[:] demand, long[:,:] scaled_cost, int n, int m, int tot, int[:,:] true_penalty):
    # Instantiate a SimpleMinCostFlow solver.
    mcf = min_cost_flow.SimpleMinCostFlow()
    # Add each arc.
    cdef int i
    cdef int j
    cdef int[:,:] arcs = np.full((n, m), -1, dtype=np.intc)
    for i in range(n):
        for j in range(m):
            if true_penalty[i, j] > 0:
                arcs[i, j] = mcf.add_arc_with_capacity_and_unit_cost(
                    i, n+j, tot, scaled_cost[i, j])
    # Add node supplies.
    cdef int[:] all_nodes_indices = np.arange(n+m, dtype=np.intc)
    cdef int[:] all_nodes_supply_demand = np.concatenate((supply, demand), dtype=np.intc)
    mcf.set_nodes_supplies(all_nodes_indices, all_nodes_supply_demand)
    # Find the minimum cost flow
    cdef int result = mcf.solve_max_flow_with_min_cost()
    # Retrieve the optimal flows
    cdef cnp.ndarray[int, ndim=2] plan = np.zeros((n, m), dtype=np.intc)
    cdef int[:,:] plan_view = plan
    if result == mcf.OPTIMAL:
    #    cdef int[:,:] solution_flows = mcf.flows()
        for i in range(n):
            for j in range(m):
                if arcs[i, j] >= 0:
                    plan_view[i, j] = mcf.flow(arcs[i, j])
        return plan
    else:
        print(f'Problems!!! result = {result}')
        raise RuntimeError('There was an issue with the mincostflow solver.')