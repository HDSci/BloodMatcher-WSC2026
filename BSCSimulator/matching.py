"""Classes for matching platform/area and algorithms."""

import numpy as np

from .antigen import Antigens, not_compatible, not_exact_match_aborhd, not_exact_match_abo, not_exact_match_rhd
from .metrics import (can_unit_be_used_to_rebalance, discount_future_demand,
                      fifo_discount, mismatching, restrict_r0_units_from_rebalancing,
                      major_antigen_substitution, sum_of_substitutions, transportation_cost,
                      young_blood_penalty, transit_constraint, IntermediateMetrics, siloed_future_demand_supply_constraint)
from .mincostflow import maxflow_mincost
from .util import list_of_permutations
from .location import DEMAND_COLUMNS, INVENTORY_COLUMNS
from .location import Inventory, Locations

MATCHES_COLUMNS = 8
"""Columns -> 0: ID demand, 1: ID supply, 2: date, 3: phenotype demand, 4: phenotype supply, 5: patient group,
6: location ID demand, 7: location ID supply"""

F_MATCHES_COLUMNS = 11
"""Columns-> 0: ID demand, 1: ID supply, 2: date, 3: phenotype demand, 4: phenotype supply, 5: patient group,
6: location ID demand, 7: sink ID demand, 8: location ID supply, 9: request due-date, 10: transit time"""

ALL_PENALTY_NAMES = ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood', 'transport_cost']
ALL_PENALTY_INDICES = {name: i for i, name in enumerate(ALL_PENALTY_NAMES)}


class MatchingArea:

    def __init__(
            self, algo=None, antigens: Antigens = None, anticipation=False, cost_weights=None,
            solver='maxflow', young_blood_constraint=True, substitution_penalty_parity=True,
            unit_moving_window=0, penalty_names:list=None, discount_future: str = None,
            expert_mismatch_risk=False, multi_stage_matching:list[bool]=None,
            multi_stage_strictness_factors:list[float]=None,
            multi_stage_constrain_future_requests:bool=False,
            random_relaxation:bool=False, siloed_future:bool=False,
            mismatching_parameters:dict=None,
            exact_matching_patient_groups:list[str]=None,
            mismatch_normalisation:bool=True,) -> None:
        self.current_date = 0
        self.matches = np.empty((0, MATCHES_COLUMNS), dtype=int)
        self.pending_requests = np.empty(shape=(0, DEMAND_COLUMNS + 1), dtype=int)  # last column = matched or not
        self.copy_of_inventory = None
        self.matching_algo = algo
        self.antigens = antigens
        self._todays_matches = np.empty((0, MATCHES_COLUMNS), dtype=int)
        """Columns -> 0: ID demand, 1: ID supply, 2: date, 3: phenotype demand, 4: phenotype supply, 5: patient group,
        6: location ID demand, 7: location ID supply"""
        self._futures_matches = np.empty((0, F_MATCHES_COLUMNS), dtype=int)
        """Columns-> 0: ID demand, 1: ID supply, 2: date, 3: phenotype demand, 4: phenotype supply, 5: patient group,
        6: location ID demand, 7: sink ID demand, 8: location ID supply, 9: request due-date, 10: transit time"""
        self._rebalance_flows = np.empty((0, F_MATCHES_COLUMNS), dtype=int)
        """Columns-> 0: ID demand, 1: ID supply, 2: date, 3: phenotype demand, 4: phenotype supply, 5: patient group,
        6: location ID demand, 7: sink ID demand, 8: location ID supply, 9: request due-date, 10: transit time"""
        self.matches_costs = None
        self.immediately_unmet_requests = np.empty(shape=(0, DEMAND_COLUMNS + 2), dtype=int)  # Penultimate column = matched or not; last column = time
        self.num_matches = 0
        self.abo_cm_combos = None
        self.abo_cm_counts = 0
        self.scd_shortages = 0
        self.all_shortages = 0
        self.pr_allo_abs = np.empty(shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        self.anticipation = anticipation
        self.forecast_units = np.empty(shape=(0, INVENTORY_COLUMNS), dtype=int)
        self.forecast_requests = np.empty(shape=(0, DEMAND_COLUMNS + 1), dtype=int)  # 5th column = matched or not
        self.fr_allo_abs = np.empty(shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        if type(cost_weights) is dict:
            penalty_names = list(cost_weights.keys()) if len(cost_weights) > 0 else None
            cost_weights = list(cost_weights.values()) if len(cost_weights) > 0 else None
        if penalty_names is None:
            penalty_names = ALL_PENALTY_NAMES
        if cost_weights is None:
            cost_weights = np.ones(len(penalty_names), dtype=int)
        elif np.abs(cost_weights).sum() == 0:
            cost_weights = np.ones(len(penalty_names), dtype=int)
        if substitution_penalty_parity:
            # Equal weight for major antigen substitution and sum of substitutions
            cost_weights[ALL_PENALTY_INDICES['substitutions']] = cost_weights[ALL_PENALTY_INDICES['usability']]
        _cost_weights = np.zeros(len(ALL_PENALTY_NAMES))
        assert len(penalty_names) == len(cost_weights), f'Length of penalty names ({len(penalty_names)}) and cost weights ({len(cost_weights)}) do not match.'
        for i, name in enumerate(penalty_names):
            _cost_weights[ALL_PENALTY_INDICES[name]] = cost_weights[i]
        _cost_weights = _cost_weights / np.abs(_cost_weights).sum()
        self.transport_matching_weights = _cost_weights
        self.ages_given_to_scd = np.empty(shape=(0, 35+1), dtype=int)
        self.solver = solver
        self.yb_constraint = young_blood_constraint
        self.abod_mm_combos = None
        self.abod_mm_counts = 0
        self.abo_mm_combos = None
        self.abo_mm_counts = 0
        self.d_mm_combos = None
        self.d_mm_counts = 0
        self.abod_mm_pat_counts = 0
        self.abo_mm_pat_counts = 0
        self.d_mm_pat_counts = 0
        self.move_window = unit_moving_window
        self.rebalance_orders = np.empty(shape=(0, DEMAND_COLUMNS + 1), dtype=int) # Last column = matched or not
        self.rebalance_orders_allo_abs = np.empty(shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        self.rebalance_orders_sources: list[np.ndarray] = []
        self._dem_to_sup_map = np.empty(0, dtype=int)
        """Location ID of demand point to location ID of primary supply point"""
        self._id_to_sink_map = np.empty(0, dtype=int)
        """Location unique ID to location sink ID"""
        self.discount_future = discount_future
        self._use_expert_risk = expert_mismatch_risk
        self.multi_stage_matching = [False, False, False] if multi_stage_matching is None or multi_stage_matching is False else multi_stage_matching
        if multi_stage_strictness_factors is None:
            self.multi_stage_strictness = [1, 1, 1]
        elif type(multi_stage_strictness_factors) is list:
            self.multi_stage_strictness = multi_stage_strictness_factors
        else:
            self.multi_stage_strictness = [multi_stage_strictness_factors] * 3
        self.constrain_future_requests = multi_stage_constrain_future_requests
        self.relaxation_randomisation = random_relaxation
        self.multi_stage_rng = np.random.default_rng(20241213)
        self.siloed_future_requests = siloed_future
        #### PATIENT SPECIFIC ATTRIBUTES #####
        self.young_blood_groups = []
        self.group_specific_risks: dict[str, dict] = {}
        self.mismatching_parameters = mismatching_parameters
        self.exact_matching_pat_groups = set(exact_matching_patient_groups) if exact_matching_patient_groups is not None else set()
        self.mismatch_normalisation = mismatch_normalisation

    def tick(self):
        self.current_date += 1

    def track_unmatched_requests(self):
        if self._remaining_requests[0].size == 0:
            return
        remaining_requests = self._remaining_requests[0]
        time = np.full(len(remaining_requests), self.current_date, dtype=int)[:, None]
        unmet_requests = np.hstack((remaining_requests, time))
        self.immediately_unmet_requests = np.vstack((self.immediately_unmet_requests, unmet_requests))
        self._remaining_requests = (None, None)

    def remove_matched_requests(self, backlog_strategy=None):
        deficit = self.pending_requests[:, 2] - self.pending_requests[:, 7]
        j = deficit > 0
        i = ~j
        pr_copy = self.pending_requests.copy()
        self._fulfilled_requests = self.pending_requests[i]
        pr_copy[j, 2] = deficit[j]
        pr_copy[j, 7] = 0
        self._remaining_requests = pr_copy[j], self.pr_allo_abs[j]
        if backlog_strategy is None or backlog_strategy == 'clear' or backlog_strategy is False:
            self.pending_requests = self.pending_requests[np.zeros(len(self.pending_requests), dtype=bool)]
            self.pr_allo_abs = self.pr_allo_abs[np.zeros(len(self.pr_allo_abs), dtype=bool)]
        elif backlog_strategy == 'keep':
            self.pending_requests = pr_copy[j]
            self.pr_allo_abs = self.pr_allo_abs[j]

    def get_inventory(self, inventory: Inventory):
        store = inventory.store.copy()
        self.copy_of_inventory = store
        
    def get_demand_details(self, demand):
        self.get_patient_groups(demand)
        self._get_bulk_demand_to_exclude_from_forecast(demand)
        self.get_usabilities(demand)
        self.unpack_mismatching_parameters()

    def get_patient_groups(self, demand: Locations):
        self.patient_groups = demand.patient_groups
        
    def _get_bulk_demand_to_exclude_from_forecast(self, demand: Locations):
        """Get bulk demand groups to exclude from forecasted demand."""
        self._bulk_groups_to_exclude = list(demand.bulk_demand_groups.values())

    def unpack_mismatching_parameters(self,):
        """Unpack mismatching parameters from the input dictionary, setting up group-specific risks where provided."""
        # Unpack default parameters
        default_mismatch_immune_risk = self.mismatching_parameters.get('mismatch_immune_risk')
        default_mismatch_expert_risk = self.mismatching_parameters.get('mismatch_expert_risk')
        default_substitution_penalty = self.mismatching_parameters.get('substitution_penalty')
        default_mismatch_mask = self.mismatching_parameters.get('mismatch_mask')
        default_subst_mask = self.mismatching_parameters.get('substitution_mask')
        # self._immuno_risk_norm = 0
        # self._mismatch_risk_norm = 0
        # Unpack group-specific parameters
        for pat_grp in self.patient_groups.keys():
            params = {pat_grp: {'immuno_risk': self.mismatching_parameters.get(f'{pat_grp}_mismatch_immune_risk', default_mismatch_immune_risk)[self.antigens.antigen_index].to_numpy().flatten(),
                                'mismatch_risk': self.mismatching_parameters.get(f'{pat_grp}_mismatch_expert_risk', default_mismatch_expert_risk)[self.antigens.antigen_index].to_numpy().flatten(),
                                'subst_immuno_risk': self.mismatching_parameters.get(f'{pat_grp}_substitution_penalty', default_substitution_penalty)[self.antigens.antigen_index].to_numpy().flatten(),
                                'mismatch_mask': self.mismatching_parameters.get(f'{pat_grp}_mismatch_mask', default_mismatch_mask)[self.antigens.antigen_index].to_numpy().flatten(),
                                'subst_mask': self.mismatching_parameters.get(f'{pat_grp}_substitution_mask', default_subst_mask)[self.antigens.antigen_index].to_numpy().flatten(),
                            }
            }
            self.group_specific_risks.update(params)
            # self._immuno_risk_norm = max(self._immuno_risk_norm, params[pat_grp]['immuno_risk'].sum())
            # self._mismatch_risk_norm = max(self._mismatch_risk_norm, params[pat_grp]['mismatch_risk'].sum())

    def get_transport_details(self, locations: Locations):
        self._transportation_costs = locations.courier_costs
        self._transit_time_data = locations.transit_time_data

    def get_location_id_mappings(self, locations: Locations):
        self._dem_to_sup_map = locations.demand_to_primary_supply_map
        """Location ID of demand point to location ID of primary supply point"""
        self._id_to_sink_map = locations.location_id_to_sink_id
        """Location unique ID to location sink ID"""

    def receive_new_requests(self, demand: np.ndarray, abs_mask: np.ndarray=None):
        demand = np.atleast_2d(demand)
        if demand.size <= 0:
            return
        if demand.shape[1] < self.pending_requests.shape[1]:
            padding_shape = (demand.shape[0], self.pending_requests.shape[1] - demand.shape[1])
            padding = np.zeros(padding_shape, dtype=int)
            demand = np.hstack((demand, padding))
        self.pending_requests = np.vstack((self.pending_requests, demand))
        if abs_mask is None:
            abs_mask = np.full((len(demand), self.pr_allo_abs.shape[1]), False)
        self.pr_allo_abs = np.vstack((self.pr_allo_abs, abs_mask))

    def get_usabilities(self, demand: Locations):
        """Get the population ABD usabilities from the "Demand"-like object or the antigens object."""
        if self.antigens.population_abd_usabilities is not None:
            self.population_abd_usabilities = self.antigens.population_abd_usabilities
        else:
            self.population_abd_usabilities = demand.population_abd_usabilities

    def get_forecasts(self, units: np.ndarray, requests: tuple[np.ndarray, np.ndarray]):
        """Get forecasted units and requests.
        
        Currently removes patient groups that are of type 'Bulk' from forecasted requests.
        
        :param np.ndarray units: Forecasted units.
        :param tuple requests: First element is a 2D numpy array of requests,
        second element is the 2-D numpy array for the alloantibody mask.
        """
        # self.forecast_units = units.copy()
        # TODO: This needs to be handled in a better way
        padding_shape_u = (units.shape[0], 1)
        padding_u = np.zeros(padding_shape_u, dtype=int)
        self.forecast_units = np.hstack((units, padding_u))
        padding_shape_r = (requests[0].shape[0], 1)
        padding_r = np.zeros(padding_shape_r, dtype=int)
        self.forecast_requests = np.hstack((requests[0], padding_r))
        self.fr_allo_abs = requests[1].copy()
        not_bulk_requests = np.isin(self.forecast_requests[:, 4], self._bulk_groups_to_exclude, invert=True)
        self.forecast_requests = self.forecast_requests[not_bulk_requests]
        self.fr_allo_abs = self.fr_allo_abs[not_bulk_requests]

    def receive_rebalance_orders(self, orders: tuple[np.ndarray, np.ndarray, list[np.ndarray]]):
        """Receive orders to rebalance inventory.
        
        :param tuple orders: First element is a 2D numpy array of orders,
        second element is the 2-D numpy array for the alloantibody mask,
        third element is a list of 1-D numpy arrays for the source location IDs.
        """
        # TODO: This needs to be handled in a better way
        padding_shape = (orders[0].shape[0], 1)
        padding = np.zeros(padding_shape, dtype=int)
        self.rebalance_orders = np.hstack((orders[0], padding))
        self.rebalance_orders_allo_abs = orders[1].copy()
        self.rebalance_orders_sources = orders[2].copy()
        
    def get_units_to_move(self) -> tuple[np.ndarray, ...]:
        """Get information on units that should be moved.
        
        Current implementation gets ids of units
        and ids of locations to move them to.
        Returns tuple of those arrays."""
        units_ids = np.empty(0, dtype=np.int64)
        new_location_ids = np.empty(0, dtype=np.int64)
        request_due_dates = np.empty(0, dtype=np.int64)
        patient_groups = np.empty(0, dtype=np.int64)
        new_location_demand_ids = np.empty(0, dtype=np.int64)
        transit_times = np.empty(0, dtype=np.int64)
        
        units_ids = np.concatenate((units_ids, self._futures_matches[:, 1], self._rebalance_flows[:, 1]))
        new_location_ids = np.concatenate((new_location_ids,
                                           self._dem_to_sup_map[self._futures_matches[:, 6]],
                                           self._rebalance_flows[:, 6]))
        request_due_dates = np.concatenate((request_due_dates, self._futures_matches[:, 9], self._rebalance_flows[:, 9]))
        patient_groups = np.concatenate((patient_groups, self._futures_matches[:, 5], self._rebalance_flows[:, 5]))
        transit_times = np.concatenate((transit_times, self._futures_matches[:, 10], self._rebalance_flows[:, 10]))
        new_location_demand_ids = self._id_to_sink_map[new_location_ids]
        return units_ids, new_location_ids, request_due_dates, patient_groups, new_location_demand_ids, transit_times

    def matching_algorithm(self, shelf_life=35, max_young_blood=14,):
        self._futures_matches = np.empty((0, F_MATCHES_COLUMNS), dtype=int)
        self._todays_matches = np.empty((0, MATCHES_COLUMNS), dtype=int)
        self._rebalance_flows = np.empty((0, F_MATCHES_COLUMNS), dtype=int)
        if self.matching_algo is None or self.matching_algo == 'default':
            return self.default_matching()
        elif self.matching_algo == 'transport':
            return self.transport_matching(shelf_life=shelf_life, max_young_blood=max_young_blood,)
        raise ValueError(f'Unknown matching algorithm: {self.matching_algo}')

    def default_matching(self):
        raise NotImplementedError('Default matching algorithm is not implemented.')

    # @jit(nopython=False)
    def transport_matching(self, shelf_life=35, max_young_blood=14, solver=None):
        matches = []
        if self.pending_requests.size == 0:
            self._todays_matches = np.array(matches)
            return matches

        # Use default solver if none provided
        if solver is None:
            solver = self.solver
        # Columns -> 0: ID, 1: phenotype, 2: units, 3: due date, 4: patient group, 5: location ID, 6: location sink ID, 7: matched or not
        reqs = self.pending_requests
        reqs_ab_mask = self.pr_allo_abs
        # Columns -> 0: ID, 1: antigen vector, 2: date bled, 3: location ID, 4: location sink ID, 5: remaining transit time
        units = self.copy_of_inventory
        num_f_units = 0  # Count of forecasted units
        # Include forecasted units if anticipation is enabled
        if self.anticipation and self.forecast_units.size > 0:
            units = np.vstack((units, self.forecast_units))
            num_f_units = len(self.forecast_units)
        num_f_reqs = 0
        # Include predicted future patient requests
        if self.anticipation and self.forecast_requests.size > 0:
            reqs = np.vstack((reqs, self.forecast_requests))
            reqs_ab_mask = np.vstack((reqs_ab_mask, self.fr_allo_abs))
            num_f_reqs = len(self.forecast_requests)
        num_r_reqs = 0
        # Include rebalance requests
        if self.rebalance_orders.size > 0:
            reqs = np.vstack((reqs, self.rebalance_orders))
            reqs_ab_mask = np.vstack((reqs_ab_mask, self.rebalance_orders_allo_abs))
            num_r_reqs = len(self.rebalance_orders)
        # Convert phenotype integer to binary antigen representation
        reqs_phen = self.antigens.convert_to_binarray(reqs[:, 1])
        units_phen = self.antigens.convert_to_binarray(units[:, 1])
        num_units = len(units_phen)
        num_reqs = len(reqs_phen)
        units_hist = np.ones(num_units, np.int64)  # Units available
        reqs_hist = reqs[:, 2].astype(np.int64)  # Units requested per patient
        sum_reqs = reqs_hist.sum()  # Total demand
        num_t_units = num_units - num_f_units  # Today's available units
        num_t_reqs = num_reqs - num_f_reqs - num_r_reqs  # Today's patient requests
        # Alloantibodies
        # Start with assumption that all antigens are positive (1 = +ve)
        reqs_abs = np.ones((reqs_phen.shape[0], reqs_phen.shape[1] - 3), dtype=int)
        # Set antigens to 0 (-ve)where patient lacks antigen AND has antibody
        abs_idx = (reqs_phen[:, 3:] == 0) & reqs_ab_mask
        reqs_abs[abs_idx] = 0
        reqs_abo_abs_phens = np.hstack((reqs_phen[:, :3], reqs_abs))

        # Calculate components of cost function
        intermediate_calcs = IntermediateMetrics()
        ###### ADDING PATIENT ######
        # Create boolean masks for each patient group to enable group-specific processing
        patient_groups = self.patient_groups
        patient_group_masks = {}
        # Patients needing exact matches
        exact_matching_patients = np.zeros(num_reqs, dtype=bool)
        # Patients needing young blood
        young_blood_patients = np.zeros(num_reqs, dtype=bool)
        # Process each patient group to set up group-specific requirements
        for group_name, group_id in patient_groups.items():
            patient_group_masks[group_name] = reqs[:, 4] == group_id
            # Checks if certain patient groups require exact antigen matching
            if group_name in self.exact_matching_pat_groups:
                exact_matching_patients |= patient_group_masks[group_name]
            # Determine which patients need young blood based on group membership
            if group_name in self.young_blood_groups:
                young_blood_patients |= patient_group_masks[group_name]

        ##### PENALTY CALCULATION SETUP ############
        w = self.transport_matching_weights

        # Initialize penalty matrices for extended matching rules
        imm = np.zeros((num_reqs, num_units))  # immune mistmatch penalties
        subst = np.zeros((num_reqs, num_units))  # substitution penalties
        # Calculate group-specific penalties for each patient group
        for group_name, group_mask in patient_group_masks.items():
            if not np.any(group_mask):
                continue
            # Retrieve group-specific or default risks
            group_risks = self.group_specific_risks[group_name]
            immuno_risk = group_risks.get('immuno_risk',)
            mismatch_risk = group_risks.get('mismatch_risk',)
            subst_immuno_risk = group_risks.get('subst_immuno_risk',)
            group_mismatch_mask = group_risks.get('mismatch_mask',)
            subst_pen_mask = group_risks.get('subst_mask',)
            # Choose expert or immuno risk
            risk = mismatch_risk if self._use_expert_risk else immuno_risk
            # Calculate penalties using mismatching function
            group_imm = mismatching(units_phen[:, group_mismatch_mask],  # Unit antigens
                                    reqs_phen[group_mask][:, group_mismatch_mask],  # Request antigens for this group
                                    np.asarray(risk)[group_mismatch_mask], # Risk weights for each antigen
                                    normalise=self.mismatch_normalisation,)
            imm[group_mask] = group_imm
            # Compute substitution penalties (if weighted)
            if w[2] != 0:
                group_subst = sum_of_substitutions(units_phen[:, subst_pen_mask],
                                                    reqs_phen[group_mask][:, subst_pen_mask],
                                                    np.asarray(subst_immuno_risk)[subst_pen_mask],)
                subst[group_mask] = group_subst
            req_abo_abs_phen_pat_group = reqs_abo_abs_phens[group_mask]
            group_compat_mask = ~group_mismatch_mask
            req_abo_abs_phen_pat_group[:, group_compat_mask] = reqs_phen[np.ix_(group_mask, group_compat_mask)]
            reqs_abo_abs_phens[group_mask] = req_abo_abs_phen_pat_group

        # Create combined ABO/antibody phenotype
        reqs_abo_abs = self.antigens.binarray_to_int(reqs_abo_abs_phens)

        ##### USABILITY AND COMPATIBILITY CALCULATIONS ############
        if w[1] == 0:
            usab_diff = np.zeros((1, 1))
        else:
            units_abod_ints = units[:, 1] >> self.antigens.vector_length - 3
            reqs_abod_ints = reqs[:, 1] >> self.antigens.vector_length - 3
            usab_diff = major_antigen_substitution(units_abod_ints, reqs_abod_ints, self.anticipation,
                                                    self.population_abd_usabilities)
        fifo = fifo_discount(shelf_life + units[:, 2] - self.current_date, shelf_life,
                             reqs[:, 3] - self.current_date,
                             young_blood_patients * self.yb_constraint)
        if w[4] == 0 or self.yb_constraint is not True:
            old_blood = np.zeros((1, 1))
        else:
            old_blood = young_blood_penalty(self.current_date - units[:, 2], max_young_blood,
                                            reqs[:, 3] - self.current_date, young_blood_patients)
        antigen_compatibility = not_compatible(reqs_abo_abs, units[:, 1][None, :], self.antigens.mask)
        abod_incompat_indices = antigen_compatibility > 0
        incompat_indices = usab_diff < 0
        usab_diff[incompat_indices] = 0
        abod_incompat = np.zeros(antigen_compatibility.shape)
        abod_incompat[abod_incompat_indices] = 1e16
        unit_available_dates = np.maximum(units[:, 2], self.current_date) + units[:, 5]
        location_transit_compatibility = transit_constraint(self._transit_time_data,
                                                            units[:, 3], reqs[:, 6],  # Using sink IDs here, should I be using primary supply IDs or even primary supply sink IDs? ¯\_(ツ)_/¯
                                                            unit_available_dates, reqs[:, 3], stepwise_results=intermediate_calcs,)
        transport_cost = np.zeros(antigen_compatibility.shape)
        if w[5] != 0:
            transport_cost[num_t_reqs:, :] = transportation_cost(units[:, 3], reqs[num_t_reqs:, 6],
                                                                self._transportation_costs,)
        if self.siloed_future_requests:  # So that future requests cannot be met with units from another SHU. No moving of units to meet future demand
            siloed_shus_constraint = siloed_future_demand_supply_constraint(num_t_reqs, num_f_reqs, stepwise_results=intermediate_calcs,)
        else:
            siloed_shus_constraint = 0
        rebalance_compatibility = np.zeros(abod_incompat.shape)
        rebalance_compatibility[(num_t_reqs + num_f_reqs):, :] = can_unit_be_used_to_rebalance(units[:, 3], self.rebalance_orders_sources)
        rebalance_compatibility[(num_t_reqs + num_f_reqs):] += restrict_r0_units_from_rebalancing(units[:, 1], )  # R0 units cannot be used for rebalancing
        
        # TODO: Separate FIFO penalty from FIFO constraint
        # TODO: Separate Young blood penalty from Young blood constraint
        core_fifo = np.abs(fifo) <= 1
        w_3_fifo = fifo * 1
        w_3_fifo[core_fifo] *= w[3]
        core_old_blood = old_blood <= young_blood_penalty(np.array([max_young_blood - 1]), max_young_blood)[0, 0]
        w_4_old_blood = old_blood * self.yb_constraint
        w_4_old_blood[core_old_blood] *= w[4]
        
        staging_policy = self.multi_stage_matching
        strictness_factors = self.multi_stage_strictness
        units_abod_ints = (units[:, 1] >> self.antigens.vector_length - 3)[None, :]
        reqs_abod_ints = (reqs[:, 1] >> self.antigens.vector_length - 3)[:, None]
        if staging_policy[0]:
            stage_one = not_exact_match_aborhd(reqs_abod_ints, units_abod_ints,)
        else:
            stage_one = np.zeros((1, 1), dtype=bool)
        if staging_policy[1]:
            stage_two = not_exact_match_abo(reqs_abod_ints, units_abod_ints, remove_rhd=True)
        else:
            stage_two = np.zeros((1, 1), dtype=bool)
        if staging_policy[2]:
            stage_three = not_exact_match_rhd(reqs_abod_ints, units_abod_ints, remove_abo=True)
        else:
            stage_three = np.zeros((1, 1), dtype=bool)
        staging_constraints = [stage_one, stage_two, stage_three]

       # Build cost matrix
        cost_matrix = w[0] * imm + w[2] * subst + w[1] * usab_diff + abod_incompat - w_3_fifo + w_4_old_blood + \
            w[5] * transport_cost + rebalance_compatibility + location_transit_compatibility + siloed_shus_constraint

        # 'Discount' allocations to future requests and rebalance orders
        if self.discount_future is not None:
            cost_matrix = discount_future_demand(cost_matrix, reqs[:, 3] - self.current_date, self.discount_future)
        
        if solver.lower() != 'maxflow' and solver.lower() != 'ortools-maxflow':
            raise ValueError('Only maxflow solvers are supported.')

        bindex_b = (cost_matrix > 1e15) | (cost_matrix < -1e15)
        cost_matrix[bindex_b] = 1e8
        bindex_s = (cost_matrix <= 100) & (cost_matrix >= -100)
        cost_matrix[bindex_s] *= 1000
        
        staged_cost_matrix = cost_matrix.copy()
        # Use combined exact matching patients
        staged_exact_patients = exact_matching_patients.copy()
        staged_units_hist = units_hist.copy()
        staged_reqs_hist = reqs_hist.copy()
        composite_staged_plan = np.zeros(cost_matrix.T.shape, dtype=int)
        all_reqs_enumerated = np.arange(num_reqs)
        staged_todays_reqs_indices = all_reqs_enumerated < num_t_reqs
        staged_todays_or_and_future_reqs_indices = staged_todays_reqs_indices  # If not constraining future requests, then this is just today's requests ELSE it is today's + future requests
        staged_todays_and_future_reqs_indices = (all_reqs_enumerated < (num_t_reqs + num_f_reqs)) | staged_todays_reqs_indices
        has_all_demand_been_met = False
        num_compatible_units = np.sum((abod_incompat + location_transit_compatibility + siloed_shus_constraint) == 0, axis=1)
        o_neg_reqs: np.ndarray = reqs_abod_ints == 0
        random_scores = self.multi_stage_rng.random(num_reqs)
        staged_fully_met_requests = np.zeros(num_reqs, dtype=bool)

        if self.constrain_future_requests:
            staged_todays_or_and_future_reqs_indices = staged_todays_and_future_reqs_indices
        randomised_relaxation = self.relaxation_randomisation

        for stage, stage_constraint, factor in zip(staging_policy, staging_constraints, strictness_factors):
            if not stage or factor == 0:
                continue
            staged_cost_matrix = cost_matrix.copy()
            if factor == 1:
                part_relaxed_constraint = stage_constraint
            else:
                scores = num_compatible_units.copy() if not randomised_relaxation else random_scores.copy()
                part_relaxed_constraint = stage_constraint.copy()
                scores[o_neg_reqs.flatten() | staged_fully_met_requests | (~staged_todays_or_and_future_reqs_indices)] = num_units
                relaxed_patients = np.argsort(scores)[:-int(factor * len(scores))]
                part_relaxed_constraint[relaxed_patients] = False
            staged_bindex_b = bindex_b | (part_relaxed_constraint & staged_exact_patients[:, None] & staged_todays_or_and_future_reqs_indices[:, None])
            staged_cost_matrix[staged_bindex_b] = 1e8

            staged_plan: np.ndarray = maxflow_mincost(staged_units_hist, staged_reqs_hist, ~staged_bindex_b.T, staged_cost_matrix.T)
            staged_requests_allocations = staged_plan.sum(axis=0)
            staged_fully_met_requests = staged_requests_allocations == staged_reqs_hist
            staged_allocations = (staged_plan > 0) & staged_fully_met_requests & staged_todays_reqs_indices & staged_exact_patients & (staged_reqs_hist > 0)
            staged_d, staged_p = np.where(staged_allocations)
            staged_units_hist[staged_d] = 0
            staged_reqs_hist[staged_p] = 0
            have_exact_patients_been_met = np.all(staged_fully_met_requests[staged_exact_patients & staged_todays_reqs_indices])
            has_all_demand_been_met = (composite_staged_plan.sum() + staged_plan.sum()) == sum_reqs

            if has_all_demand_been_met:
                composite_staged_plan += staged_plan
                break
            composite_staged_plan[staged_allocations] += staged_plan[staged_allocations]
            if have_exact_patients_been_met:
                break
        if not has_all_demand_been_met:
            plan = maxflow_mincost(staged_units_hist, staged_reqs_hist, ~bindex_b.T, cost_matrix.T)
            composite_staged_plan += plan

        todays_plan = composite_staged_plan[:num_t_units, :num_t_reqs].T
        futures_plan = composite_staged_plan[:, num_t_reqs:num_t_reqs+num_f_reqs].T
        rebalance_plan = composite_staged_plan[:, num_t_reqs+num_f_reqs:].T
        assert not np.any(
            todays_plan[abod_incompat_indices[:num_t_reqs, :num_t_units]] > 0)

        pj, di = np.where(todays_plan > 0)
        pjj2, dii2 = np.where(futures_plan > 0)
        pj3, di3 = np.where(rebalance_plan > 0)

        times = np.full(len(di), self.current_date, np.int64)[:, None]
        self.pending_requests[:num_t_reqs, 7] = todays_plan.sum(axis=1)
        matches = np.hstack((reqs[pj, 0:1], units[di, 0:1], times, reqs[pj, 1:2],
                             units[di, 1:2], reqs[pj, 4:5], reqs[pj, 5:6], units[di, 3:4]))
        self._todays_matches = matches
        # Future Matches
        f_reqs = reqs[num_t_reqs:num_t_reqs+num_f_reqs]
        ftimes = np.full(len(dii2), self.current_date, np.int64)[:, None]
        f_transit_times = intermediate_calcs.transit_times[num_t_reqs:num_t_reqs+num_f_reqs, :]
        fmatches = np.hstack((f_reqs[pjj2, 0:1], units[dii2, 0:1], ftimes,
                              f_reqs[pjj2, 1:2], units[dii2, 1:2],
                              f_reqs[pjj2, 4:5],
                              f_reqs[pjj2, 5:6], f_reqs[pjj2, 6:7], units[dii2, 3:4],
                              f_reqs[pjj2, 3:4], f_transit_times[pjj2, dii2][:, None]))
        units_arriving_exactly_on_time_future = intermediate_calcs.units_arriving_exactly_on_time[num_t_reqs:num_t_reqs+num_f_reqs, :]
        to_move_now = units_arriving_exactly_on_time_future[pjj2, dii2]  # TODO: Incorporate a moving time buffer potentially. So, if unit arrives X days before, it can be moved now.
        from_different_location = fmatches[:, 7] % fmatches[:, 8] != 0
        not_already_in_transit = units[dii2, 5] == 0
        not_future_units = units[dii2, 2] <= self.current_date  # TODO: This is not sufficient anymore because future units could be units in manufacturing that are not yet in inventory but obviously have a bled date in the past. Need a more robust way to identify future units that are not yet in inventory.
        self._futures_matches = np.vstack((self._futures_matches,
                                           fmatches[to_move_now & from_different_location & not_already_in_transit & not_future_units, :]))

        r_reqs = reqs[num_t_reqs+num_f_reqs:]
        rtimes = np.full(len(di3), self.current_date, np.int64)[:, None]
        r_transit_times = intermediate_calcs.transit_times[num_t_reqs+num_f_reqs:, :]
        rmatches = np.hstack((r_reqs[pj3, 0:1], units[di3, 0:1], rtimes,
                              r_reqs[pj3, 1:2], units[di3, 1:2],
                              r_reqs[pj3, 4:5],
                              r_reqs[pj3, 5:6], r_reqs[pj3, 6:7], units[di3, 3:4],
                              r_reqs[pj3, 3:4], r_transit_times[pj3, di3][:, None]))
        # Hard constraint to ensure future units cannot be used for rebalancing
        not_future_units = units[di3, 2] <= self.current_date  # TODO: This is not sufficient anymore because future units could be units in manufacturing that are not yet in inventory but obviously have a bled date in the past. Need a more robust way to identify future units that are not yet in inventory.
        self._rebalance_flows = np.vstack((self._rebalance_flows, rmatches[not_future_units, :]))

       # Measurements:
        self._measure_abod_crossmatch(di, pj, units_phen, reqs_phen)
        self._measure_abod_mixed_match_subsititutions(di, pj, units_phen, reqs_phen)
        # Measure the age distribution of the units given to SCD patients
        self._measure_ages_given_to_scd(di, units, True)

        return matches

# TODO: Needs to be updated to deal with patient groups (ids)
    def update_matches(self) -> np.ndarray:
        """Update the matches with the matches from the current day.
        
        Adds the matches from the current day to the matches from previous days.
        Also updates the number of matches and shortages.
        
        :return np.ndarray: The updated record of matches.
        """
        patient_groups = self.patient_groups
        if self._todays_matches.size > 0:
            _todays_matches = self._todays_matches
            self.matches = np.vstack((self.matches, _todays_matches))
            self.num_matches += len(_todays_matches)
            self.scd_shortages += self.pending_requests[self.pending_requests[:, 4] == patient_groups['SCD'], 2].sum() - len(
                _todays_matches[_todays_matches[:, 5] == patient_groups['SCD']])
            self.all_shortages += self.pending_requests[:, 2].sum() - len(self._todays_matches)
        else:
            self.scd_shortages += self.pending_requests[self.pending_requests[:, 4] == patient_groups['SCD'], 2].sum()
            self.all_shortages += self.pending_requests[:, 2].sum()
        return self.matches

    def warmup_clear(self):
        self.matches = np.empty((0, MATCHES_COLUMNS), dtype=int)
        self.pending_requests = np.empty(shape=(0, DEMAND_COLUMNS + 1), dtype=int)  # last column = matched or not
        self.pr_allo_abs = np.empty(shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        # self.immediately_unmet_requests = np.empty(shape=(0, DEMAND_COLUMNS + 2), dtype=int)
        self.num_matches = 0
        self.scd_shortages = 0
        self.all_shortages = 0
        self.abod_mm_combos = None
        self.abod_mm_counts = 0
        self.abo_mm_combos = None
        self.abo_mm_counts = 0
        self.d_mm_combos = None
        self.d_mm_counts = 0
        self.abod_mm_pat_counts = 0
        self.abo_mm_pat_counts = 0
        self.d_mm_pat_counts = 0

    def clear_forecasts(self):
        self.forecast_units = np.empty(shape=(0, INVENTORY_COLUMNS), dtype=int)
        self.forecast_requests = np.empty(shape=(0, DEMAND_COLUMNS + 1), dtype=int)  # last column = matched or not
        self.fr_allo_abs = np.empty(shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        
    def clear_rebalance_orders(self):
        self.rebalance_orders = np.empty(shape=(0, DEMAND_COLUMNS + 1), dtype=int)
        self.rebalance_orders_allo_abs = np.empty(shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        self.rebalance_orders_sources = []

    def push_update_to_inventory(self, inventory):
        if self._todays_matches.size <= 0:
            return
        i = np.isin(self.copy_of_inventory[:, 0], self._todays_matches[:, 1], assume_unique=True)
        _matched_units = self.copy_of_inventory[i]
        inventory.remove_from_store(_matched_units)

    def measure_mismatches(self, demand_phens, supply_phens, patient_groups,):
        mismatched = supply_phens > demand_phens
        results = {group_name: mismatched[patient_groups == id] for group_name, id in self.patient_groups.items()}
        results.update({'total': mismatched})
        return results

    def measure_cumulative_alloimmunisation(self, mismatched, patient_groups,):
        # cumulative alloimmunization per patient group
        results = {group: mismatched[patient_groups == id, 3:].sum(axis=0) * self.antigens.xA_rate 
                   for group, id in self.patient_groups.items()}
        # overall cumulative alloimmunization (total across all patients)
        results.update({'total': mismatched[:, 3:].sum(axis=0) * self.antigens.xA_rate})
        return results

    def measure_substitutions(self, demand_phens, supply_phens, patient_groups=None):
        subs = supply_phens < demand_phens
        results = {group: subs[patient_groups == id] for group, id in self.patient_groups.items()}
        # overall substitutions (total across all patients)
        results.update({'total': subs})
        return results

    def _measure_abod_crossmatch(self, i, j, units_phen, reqs_phen, remove_non_scd_demand=True):
        if remove_non_scd_demand:
            i = i[self._todays_matches[:, 5] == self.patient_groups['SCD']]
            j = j[self._todays_matches[:, 5] == self.patient_groups['SCD']]

        dons = units_phen[i, :3]
        pats = reqs_phen[j, :3]
        phens_joined = np.hstack((dons, pats))
        if self.abo_cm_combos is None:
            combs = np.array(list_of_permutations([(0, 1)] * 6))
            self.abo_cm_combos = combs
        else:
            combs = self.abo_cm_combos
        cm_count = np.zeros(len(combs))
        unique, counts = np.unique(phens_joined, axis=0, return_counts=True)
        for i, u in enumerate(unique):
            cm_count[(combs == u).all(axis=1)] += counts[i]
        self.abo_cm_counts += cm_count

    def _measure_abod_mixed_match_subsititutions(
            self, i: np.ndarray, j: np.ndarray, units_phen: np.ndarray, reqs_phen: np.ndarray):
        # Remove non-SCD patients
        i = i[self._todays_matches[:, 5] == self.patient_groups['SCD']]
        j = j[self._todays_matches[:, 5] == self.patient_groups['SCD']]
        
        # Measure number of times an abod/abo/d substitution was done
        dons = units_phen[i, :3]
        pats = reqs_phen[j, :3]
        # Concatenate ABOD phenotypes of donors with patients'
        abod_phens_joined = np.hstack((dons, pats))
        # Instantiate arrays that define combinations of substitutions
        if self.abod_mm_combos is None:
            abod_combs = np.array(list_of_permutations([(0, 1)] * 6))
            abo_combs = np.array(list_of_permutations([(0, 1)] * 4))
            d_combs = np.array(list_of_permutations([(0, 1)] * 2))
            self.abod_mm_combos = abod_combs
            self.abo_mm_combos = abo_combs
            self.d_mm_combos = d_combs
        else:
            abod_combs = self.abod_mm_combos
            abo_combs = self.abo_mm_combos
            d_combs = self.d_mm_combos
        # Instantiate arrays that count the number of substitutions
        abod_mm_count = np.zeros(len(abod_combs))
        abo_mm_count = np.zeros(len(abo_combs))
        d_mm_count = np.zeros(len(d_combs))
        # Count the number of substitutions for each unique combination
        unique, counts = np.unique(abod_phens_joined, axis=0, return_counts=True)
        for k, u in enumerate(unique):
            abod_mm_count[(abod_combs == u).all(axis=1)] += counts[k]
            abo_mm_count[(abo_combs == u[[0, 1, 3, 4]]).all(axis=1)] += counts[k]
            d_mm_count[(d_combs == u[[2, 5]]).all(axis=1)] += counts[k]
        self.abod_mm_counts += abod_mm_count
        self.abo_mm_counts += abo_mm_count
        self.d_mm_counts += d_mm_count
        
        # Measure how many patients received an abod/abo/d substitution
        unique_patients = np.unique(j)
        abod_mm_pat_count = 0
        abo_mm_pat_count = 0
        d_mm_pat_count = 0
        for k in unique_patients:
            pat_d_type = reqs_phen[k, 2]
            pat_abo_type = reqs_phen[k, :2].dot([2, 1])
            units_given_indices = k == j
            units_given_d_type = dons[units_given_indices, 2]
            units_given_abo_type = dons[units_given_indices, :2].dot([2, 1])
            d_mm_pat_count += np.any(units_given_d_type < pat_d_type) * 1
            abo_mm_pat_count += np.any(units_given_abo_type < pat_abo_type) * 1
            abod_mm_pat_count += np.any((units_given_d_type < pat_d_type) & (units_given_abo_type < pat_abo_type)) * 1
        self.abod_mm_pat_counts += abod_mm_pat_count
        self.abo_mm_pat_counts += abo_mm_pat_count
        self.d_mm_pat_counts += d_mm_pat_count

    def _measure_ages_given_to_scd(self, i, units, remove_non_scd_demand=True):
        if remove_non_scd_demand:
            i = i[self._todays_matches[:, 5] == self.patient_groups['SCD']]
        ages = self.current_date - units[i, 2] + 1
        ages_hist = np.bincount(ages, minlength=35+1)
        self.ages_given_to_scd = np.vstack((self.ages_given_to_scd, ages_hist))
