import os
import time
import traceback

import numpy as np
from multiprocess import Pool
from tqdm import tqdm

from .antigen import Antigens
from .matching import MatchingArea
from .location import Demand, Inventory, Locations, Supply


class SimulationManager:

    def __init__(self, antigens, demand, supply, matching, inventory, warm_up, horizon, cool_down,
                 replications: int = 100, seed=0xBADBEEF, precomputed_infolder=None, future_demand=False,
                 future_supply=False, forecasting=None, precompute_outfolder=None,
                 locations=None, rebalance_stock=False, rebalance_stock_days=None,
                 rebalance_lead_time: int = 1) -> None:
        self.n = replications
        self.antigens = antigens
        self.demand = demand
        self.supply = supply
        self.matching = matching
        self.inventory = inventory
        self.locations = locations
        self.warm_up = warm_up
        self.horizon = horizon
        self.cool_down = cool_down
        self.simulations: list[Simulation] = []
        self.seed = seed
        self.seeds = []
        self.failures = {'num': 0, 'seeds': []}
        self.precompute_infolder = precomputed_infolder
        self.forecast_demand = future_demand
        self.forecast_supply = future_supply
        self.forecasting = forecasting
        self.precompute_outfolder = precompute_outfolder
        self.rebalance_stock = rebalance_stock
        self.rebalance_stock_days = rebalance_stock_days
        self.rebalance_lead_time = rebalance_lead_time

    def do_simulations(self):
        _seed = self.seed
        for i in tqdm(range(self.n)):
            if i != 0:
                _seed = _seed + 17
            self.seeds.append(_seed)
            rng = np.random.default_rng(_seed)
            rngs = {k: np.random.default_rng(_seed) for k in ['supply', 'demand', 'manufacturing', 'manufacturing_routing',]}
            if self.locations is not None:
                ingredients = [self.demand(), self.supply(), self.matching(), self.inventory(),
                               self.locations(rngs, None),]
                clocks = ingredients[2:5]
            else:
                raise ValueError('Locations must be provided for simulations.')
            timing = [self.warm_up, self.horizon, self.cool_down]
            sim = Simulation(self.antigens, *ingredients, *timing, clocks, rng, self.forecast_demand,
                             self.forecast_supply, self.forecasting, rngs=rngs,
                             rebalance_stock=self.rebalance_stock, rebalance_stock_days=self.rebalance_stock_days,
                             rebalance_lead_time=self.rebalance_lead_time, i=i)
            sim.computed_vars_file = None if self.precompute_infolder is None else os.path.join(
                self.precompute_infolder, f'{i:05d}_{_seed:#X}_randvars.npz')
            sim.simulate()
            self.simulations.append(sim)
        self._get_supply_location_ids()

    def do_simulations_parallel(self, workers):
        _seed = self.seed
        _to_simulate = []
        for i in range(self.n):
            if i != 0:
                _seed = _seed + 17
            self.seeds.append(_seed)
            rng = np.random.default_rng(_seed)
            rngs = {k: np.random.default_rng(_seed) for k in ['supply', 'demand', 'manufacturing', 'manufacturing_routing',]}
            if self.locations is not None:
                ingredients = [self.demand(), self.supply(), self.matching(), self.inventory(),
                               self.locations(rngs, None),]
                clocks = ingredients[2:5]
            else:
                raise ValueError('Locations must be provided for simulations.')
            timing = [self.warm_up, self.horizon, self.cool_down]
            sim = Simulation(self.antigens, *ingredients, *timing, clocks, rng, self.forecast_demand,
                             self.forecast_supply, self.forecasting, rngs=rngs,
                             rebalance_stock=self.rebalance_stock, rebalance_stock_days=self.rebalance_stock_days,
                             rebalance_lead_time=self.rebalance_lead_time, i=i)
            sim.computed_vars_file = None if self.precompute_infolder is None else os.path.join(
                self.precompute_infolder, f'{i:05d}_{_seed:#X}_randvars.npz')
            _to_simulate.append(sim)

        p = Pool(workers)
        _simulated = list(tqdm(p.imap_unordered(simulate, _to_simulate, chunksize=1), total=self.n))
        p.close()
        p.join()
        p.terminate()
        # Sort _simulated by the attribute 'i' of each Simulation object
        _simulated.sort(key=lambda sim: sim.i)
        self.simulations = _simulated
        self._get_supply_location_ids()
    
    def _get_supply_location_ids(self):
        self.supply_location_ids = self.simulations[0].locations.supply_locations_ids

    def _pre_compute(self, i, seed):
        rng = np.random.default_rng(seed)
        rngs = {k: np.random.default_rng(seed) for k in ['supply', 'demand', 'manufacturing', 'manufacturing_routing',]}
        if self.locations is not None:
            ingredients = [self.demand(), self.supply(), self.matching(), self.inventory(),
                           self.locations(rngs, None),]
            clocks = ingredients[2:5]
        else:
            raise ValueError('Locations must be provided for precomputing.')
        timing = [self.warm_up, self.horizon, self.cool_down]
        sim = Simulation(self.antigens, *ingredients, *timing, clocks, rng, rngs=rngs, i=i)
        sim.pre_compute_random_vars()
        time.sleep(0.001)

        filename = f"{i:05d}_{seed:#X}_randvars"
        filename = os.path.join(self.precompute_outfolder, filename)
        np.savez_compressed(filename, **sim.precomputed_vars)
        del sim
        return filename

    def do_precompute(self, workers=1):
        _seed = self.seed
        args = []
        if workers > 1:
            for i in range(self.n):
                if i != 0:
                    _seed = _seed + 17
                self.seeds.append(_seed)
                args.append((i, _seed))

            p = Pool(workers)
            outs = p.starmap(self._pre_compute, args, self.n // workers)
            p.close()
            p.join()
            p.terminate()
        else:
            for i in tqdm(range(self.n)):
                if i != 0:
                    _seed = _seed + 17
                self.seeds.append(_seed)
                outs = self._pre_compute(i, _seed)
        return self.simulations

    def statistics(self):
        mismatch = {}
        allo = {}
        subs = {}
        scd_shorts = []
        stocks = []
        stocks_loc = []
        avail_stocks = []
        abo_cm = []
        abod_mm = []
        fails = []
        objs = []
        ages = []
        locs_ages = []
        phen_ages = []
        loc_phen_ages = []
        scd_ages = []
        all_shorts = []
        pats_subs = []
        comp_times = []
        ss_xA = []
        moves = {}
        unmet_requests = {}
        expiry_records = {}
        matches = {}
        for sim in self.simulations:
            i = sim.i
            if sim.failed:
                fails.append(self.seeds[i])
                continue
            sim.final_statistics()
            mismatch.update({key: mismatch.get(key, []) + [sim.stats['mismatches'][key]] for key in sim.stats['mismatches'].keys()})
            allo.update({key: allo.get(key, []) + [sim.stats['cum_allo'][key]] for key in sim.stats['cum_allo'].keys()})
            subs.update({key: subs.get(key, []) + [sim.stats['substitutions'][key]] for key in sim.stats['substitutions'].keys()})
            scd_shorts.append(sim.stats['scd_shortages'])
            stocks.append(sim.stats['stocks'])
            stocks_loc.append(sim.stats['stocks_loc'])
            avail_stocks.append(sim.stats['avail_stocks'])
            abo_cm.append(sim.stats['abo_cm'])
            abod_mm.append(sim.stats['abod_mm'])
            pats_subs.append(sim.stats['pats_mm_counts'])
            objs.append(
                (sim.stats['cum_allo']['total'].sum(),
                 sim.stats['scd_shortages'],
                 sim.stats['expiries'],
                 sim.stats['all_shortages'],
                 sim.stats['moves'].shape[0],
                 *[np.sum(sim.stats['moves'][:, 8] == pat_grp_code) for pat_grp_code in sim.stats['pat_grp_codes']],
                 sim.stats['o_type_stocks'][0],     # O-
                 sim.stats['o_type_stocks'][1],     # O+
                 sim.stats['o_type_stocks'][2],     # O- plus O+
                 sim.stats['pats_mm_counts'][0],    # D
                 sim.stats['pats_mm_counts'][1],    # ABO
                 sim.stats['pats_mm_counts'][2],    # ABOD
                 )
            )
            ages.append(sim.stats['stocks_age'])
            locs_ages.append(sim.stats['stocks_locs_age'])
            phen_ages.append(sim.stats['stocks_pheno_age'])
            loc_phen_ages.append(sim.stats['stocks_locs_pheno_age'])
            scd_ages.append(sim.stats['scd_unit_ages'])
            all_shorts.append(sim.stats['all_shortages'])
            comp_times.append(sim.computation_time)
            ss_xA.append(sim.stats['steady_state_xA'])
            moves.update({f'{i:05d}_{self.seeds[i]:#X}': sim.stats['moves']})
            unmet_requests.update({f'{i:05d}_{self.seeds[i]:#X}': sim.stats['unmet_requests']})
            expiry_records.update({f'{i:05d}_{self.seeds[i]:#X}': sim.stats['expiry_records']})
            matches.update({f'{i:05d}_{self.seeds[i]:#X}': sim.stats['matches']})
            del sim
        n = len(objs)
        self.simulations = []  # Free memory
        self.mismatches = {key: (np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(n)) for key, arr in mismatch.items()}
        self.allo = {key: (np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(n)) for key, arr in allo.items()}
        self.subs = {key: (np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(n)) for key, arr in subs.items()}
        self.scd_shorts = np.mean(scd_shorts), np.std(scd_shorts) / np.sqrt(n)
        self.stocks = np.mean(stocks, axis=0), np.std(stocks, axis=0) / np.sqrt(n)
        self.stocks_loc = np.mean(stocks_loc, axis=0), np.std(stocks_loc, axis=0) / np.sqrt(n)
        self.avail_stocks = np.mean(avail_stocks, axis=0), np.std(avail_stocks, axis=0) / np.sqrt(n)
        self.abo_cm = np.mean(abo_cm, axis=0), np.std(abo_cm, axis=0) / np.sqrt(n)
        self.abod_mm = np.mean(abod_mm, axis=0), np.std(abod_mm, axis=0) / np.sqrt(n)
        self.pats_subs = np.mean(pats_subs, axis=0), np.std(pats_subs, axis=0) / np.sqrt(n)
        self.objs = np.array(objs)
        self.ages = np.mean(ages, axis=0), np.std(ages, axis=0) / np.sqrt(n)
        self.locs_ages = np.mean(locs_ages, axis=0), np.std(locs_ages, axis=0) / np.sqrt(n)
        self.phen_ages = np.mean(phen_ages, axis=0), np.std(phen_ages, axis=0) / np.sqrt(n)
        self.loc_phen_ages = np.mean(loc_phen_ages, axis=0), np.std(loc_phen_ages, axis=0) / np.sqrt(n)
        self.scd_ages = np.mean(scd_ages, axis=0), np.std(scd_ages, axis=0) / np.sqrt(n)
        self.all_shorts = np.mean(all_shorts), np.std(all_shorts) / np.sqrt(n)
        self.steady_state_xA = np.mean(ss_xA, axis=0), np.std(ss_xA, axis=0) / np.sqrt(n)
        self.computation_times = np.array(comp_times)
        self.moves = moves
        self.unmet_requests = unmet_requests
        self.expiry_records = expiry_records
        self.matches = matches
        self.failures.update({'num': self.n - n, 'seeds': fails})


class Simulation:

    def __init__(self, antigens: Antigens, demand: Demand, supply: Supply, matching: MatchingArea,
                 inventory: Inventory, locations: Locations,
                 warm_up: int, horizon: int, cool_down: int,
                 clocks=None, rng=None, forecast_demand=False, forecast_supply=False, forecasting=dict(),
                 rngs=dict(), rebalance_stock: bool = False, rebalance_stock_days: list = None,
                 rebalance_lead_time: int = 1, i: int = 0,) -> None:
        self.clocks = clocks if clocks is not None else [demand, supply, matching, inventory]
        self.warm_up = warm_up
        self.horizon = horizon
        self.cool_down = cool_down
        self.rng = rng if rng is not None else np.random.default_rng()
        self.antigens = antigens
        self.demand = locations
        self.supply = locations
        self.locations = locations
        self.matching = matching
        self.inventory = inventory
        self.manufacturing_routing = locations
        self.time = 0
        self.stats = None
        self.failed = False
        self.precomputed_vars = None
        self.computed_vars_file = None
        self._loaded_vars = None
        self._loaded_vars_strides = None
        self._forecast_demand = forecast_demand
        self._forecast_supply = forecast_supply
        forecasting = forecasting if isinstance(forecasting, dict) else dict()
        self.forecast_supply_days = forecasting.get('units_days', 1)
        self.forecast_supply_shows = forecasting.get('units_shows', 1)
        self.forecast_demand_days = forecasting.get('requests_days', 1)
        self.forecast_demand_shows = forecasting.get('requests_shows', 1)
        self._forecast_rng = np.random.default_rng(20220228)
        self.rngs: dict = rngs
        self.computation_time = 0
        self.rebalance_stock = rebalance_stock
        self.rebalance_stock_days = rebalance_stock_days if rebalance_stock_days is not None else [6,]
        self.rebalance_lead_time = rebalance_lead_time  # Days to allow for rebalanced stock to arrive
        self.daily_steady_state_xA = []
        self.i = i

    def simulate(self):
        start_comp_time = time.time()
        try:
            self._open_loaded_vars()
            start_demand = False
            while self.time < self.horizon:
                self.tick()
                if self.time == 1:
                    self.inventory.initialise_inventory(self.locations)
                    self.manufacturing_routing.initialise_routing(self.inventory)
                    self.matching.get_demand_details(self.demand)
                    self.matching.get_location_id_mappings(self.locations)
                    self.matching.get_transport_details(self.locations)
                donated_units = self.supply.supply(rng=self.rngs.get('supply', self.rng))
                self.locations.send_on_pathway(donated_units, self.inventory, rng=self.rngs.get('manufacturing_routing', self.rng))
                routed_available_units = self.locations.receive_from_pathway()
                self.inventory.add_to_store(routed_available_units)
                self.inventory.measure_stock(locations=self.locations)
                if not start_demand:
                    start_demand = len(self.inventory.store) >=  self.inventory.inventory_size * 0.95

                new_requests, abs_mask = self.demand.demand(rng=self.rngs.get('demand', self.rng), skip_demand=not start_demand)
                self.matching.receive_new_requests(new_requests, abs_mask)
                self.matching.get_inventory(self.inventory)
                forecasts = self.get_n_days_forecast(skip=not start_demand)
                forecasts = self._randomise_forecast(forecasts)
                if forecasts is not None:
                    self.matching.get_forecasts(*forecasts)
                if self.rebalance_stock and start_demand and ((self.time - 1) % 7) in self.rebalance_stock_days:
                    rebalance_orders = self.inventory.request_rebalance_of_stock(self.locations, self.rebalance_lead_time)
                    if rebalance_orders is not None:
                        self.matching.receive_rebalance_orders(rebalance_orders)
                self.matching.matching_algorithm(shelf_life=self.inventory.shelf_life)
                self.matching.update_matches()
                self._measure_steady_state_xA()
                self.matching.push_update_to_inventory(self.inventory)
                self.matching.remove_matched_requests()
                self.matching.clear_forecasts()
                self.matching.clear_rebalance_orders()
                
                ## For testing accuracy of precomputed variables ##
                # if self.time == self.horizon:
                #     np.savez('scratch/manual_tests/forecasting/last_day_forecast.npz',
                #                 donated_units=donated_units, routed_available_units=routed_available_units,
                #                 new_requests=new_requests, abs_mask=abs_mask)
                ###############################################
                self.matching.track_unmatched_requests()
                self.inventory.remove_expired_units()
                self.inventory.move_units(*self.matching.get_units_to_move(), record=True)
                if self.warm_up == self.time:
                    self.matching.warmup_clear()
                    self.inventory.warmup_clear()
        except RuntimeError as e:
            self.failed = True
            traceback.print_exception(e)
        finally:
            self._close_loaded_vars()
        self.computation_time = time.time() - start_comp_time

    def tick(self):
        self.time += 1
        for clock in self.clocks:
            clock.tick()
            
    def _measure_steady_state_xA(self):
        todays_matches = self.matching.matches[self.matching.matches[:, 2] == self.time]
        if todays_matches.shape[0] == 0:
            self.daily_steady_state_xA.append(np.zeros(self.antigens.vector_length-3))
            return
        matched_demand = self.antigens.convert_to_binarray(todays_matches[:, 3])
        matched_supply = self.antigens.convert_to_binarray(todays_matches[:, 4])
        mismatches = self.matching.measure_mismatches(matched_demand, matched_supply, todays_matches[:, 5])
        cum_allo = self.matching.measure_cumulative_alloimmunisation(mismatches['total'], todays_matches[:, 5])
        self.daily_steady_state_xA.append(cum_allo['total'])
        
    def final_statistics(self):
        """
        Uses historical matches to measure mismatches, alloimmunisations, and substitutions.
        Assigns the results to `self.stats` dictionary.
        
        Full list of statistics:
        - mismatches: total number of mismatches per antigen
        - cum_allo: cumulative expected alloimmunisation per antigen
        - substitutions: mean number of substitutions per antigen
        - scd_shortages: number of shortages for SCD patients
        - stocks: inventory levels for major blood groups and watched phenotypes
        - stocks_loc: inventory levels for major blood groups and watched phenotypes at each location
        - abo_cm: number of units from each major blood group (ABOD) given to each major blood group
        - expiries: number of units expired
        - stocks_age: age distribution of the inventory over the simulation
        - stocks_locs_age: age distribution of the inventory at each location over the simulation
        - stocks_pheno_age: age distribution of the inventory over the simulation for each watched phenotype
        - stocks_locs_pheno_age: age distribution of the inventory at each location over the simulation for each watched phenotype
        - scd_unit_ages: age distribution of units given to SCD patients
        - all_shortages: number of shortages for all patients (SCD + dummy demand)
        - o_type_stocks: inventory levels for O-, O+, and O- plus O+
        - abod_mm: number of units from each major blood group (ABOD) given to each major blood group (SCD only)
        - pats_mm_counts: number of patients that received at least one D/ABO/ABOD substitution
        - moves: record of all moved units
        - unmet_requests: record of requests not immediately met
        - expiry_records: record of all expired units
        """
        matched_demand = self.antigens.convert_to_binarray(self.matching.matches[:, 3])
        matched_supply = self.antigens.convert_to_binarray(self.matching.matches[:, 4])
        patient_groups = self.matching.matches[:, 5]
        mismatches = self.matching.measure_mismatches(matched_demand, matched_supply, patient_groups)
        cum_allo = self.matching.measure_cumulative_alloimmunisation(mismatches['total'], patient_groups)
        substitutions = self.matching.measure_substitutions(matched_demand, matched_supply, patient_groups)
        scd_shortages = self.matching.scd_shortages
        all_shortages = self.matching.all_shortages
        stocks = np.array(self.inventory.stock_levels)
        stocks_loc = np.array(self.inventory.stock_levels_locations_raw)
        avail_stocks = np.array(self.inventory.available_stock_levels)
        o_type_stocks = self.inventory.mean_O_type_stock(start=self.warm_up, end=self.horizon - self.cool_down)
        o_type_stocks = np.append(o_type_stocks, o_type_stocks.sum())
        stocks_age = self.inventory.age_distribution
        stocks_locs_age = self.inventory.age_distribution_locs
        stocks_pheno_age = [np.array(dist) for dist in self.inventory.pheno_age_dist]
        stocks_locs_pheno_age = [np.array(loc_dist) for loc_dist in self.inventory.pheno_age_dist_loc]
        expiries = self.inventory.expired.shape[0]
        abo_cm = self.matching.abo_cm_counts
        scd_unit_ages = self.matching.ages_given_to_scd
        abod_mm = self.matching.abod_mm_counts
        pats_mm_counts = [self.matching.d_mm_pat_counts,
                          self.matching.abo_mm_pat_counts,
                          self.matching.abod_mm_pat_counts]
        moves = self.inventory.recorded_moves
        unmet_requests = self.matching.immediately_unmet_requests
        expiry_records = self.inventory.expired
        matches = self.matching.matches
        steady_state_xA = np.array(self.daily_steady_state_xA)
        pat_grp_codes = np.unique(list(self.demand.patient_groups.values()))
        self.stats = dict(mismatches={g: mm.sum(axis=0) for g, mm in mismatches.items()}, cum_allo=cum_allo,
                          substitutions={g: sb.sum(axis=0) for g, sb in substitutions.items()},
                          scd_shortages=scd_shortages,
                          stocks=stocks, abo_cm=abo_cm,
                          expiries=expiries, stocks_age=stocks_age,
                          stocks_pheno_age=stocks_pheno_age,
                          stocks_locs_pheno_age=stocks_locs_pheno_age,
                          scd_unit_ages=scd_unit_ages,
                          all_shortages=all_shortages, o_type_stocks=o_type_stocks,
                          abod_mm=abod_mm, pats_mm_counts=pats_mm_counts,
                          moves=moves, stocks_loc=stocks_loc, stocks_locs_age=stocks_locs_age,
                          unmet_requests=unmet_requests, expiry_records=expiry_records,
                          matches=matches, avail_stocks=avail_stocks,
                          steady_state_xA=steady_state_xA, pat_grp_codes=pat_grp_codes,)
        

    def pre_compute_random_vars(self):
        starting_inventory = None
        units = []
        requests = []
        requests_Abs = []
        strides = []
        donations = []
        start_demand = False
        while self.time < self.horizon:
            self.tick()
            if self.time == 1:
                self.inventory.initialise_inventory(self.locations)
                self.manufacturing_routing.initialise_routing(self.inventory)
            donated_units = self.supply.supply(rng=self.rngs.get('supply', self.rng))
            self.locations.send_on_pathway(donated_units, self.inventory, rng=self.rngs.get('manufacturing_routing', self.rng))
            routed_available_units = self.locations.receive_from_pathway()
            units.append(routed_available_units)
            if not start_demand:
                self.inventory.add_to_store(routed_available_units)  # Only add to inventory before demand starts so that it does not get too large
                start_demand = len(self.inventory.store) >= self.inventory.inventory_size * 0.95
            new_requests, abs_mask = self.demand.demand(rng=self.rngs.get('demand', self.rng), skip_demand=not start_demand)
            requests.append(new_requests)
            requests_Abs.append(abs_mask)
            donations.append(donated_units[:, [0, 3]])  # ID and location ID of original donation point only
            strides.append((len(routed_available_units), len(new_requests)))
        self.precomputed_vars = {'start_inventory': self.inventory.store[:0], 'units': np.vstack(units),
                                 'requests': np.vstack(requests), 'requests_Abs': np.vstack(requests_Abs),
                                 'strides': np.vstack(strides), 'donations': np.vstack(donations)}

    def _open_loaded_vars(self):
        if self.computed_vars_file is None:
            return
        temp = np.load(self.computed_vars_file)
        if temp['units'].shape[0] < 1.3e6:
            self._loaded_vars = {k: v.copy() for k, v in temp.items()}
            temp.close()
        else:
            self._loaded_vars = temp
        self._loaded_vars_strides: np.ndarray = self._loaded_vars['strides'].cumsum(axis=0)

    def _close_loaded_vars(self):
        if self._loaded_vars is not None:
            del self._loaded_vars_strides
            try:
                self._loaded_vars.close()
            except AttributeError:
                pass
            finally:
                self._loaded_vars = None

    def get_n_days_forecast(self, skip=False) -> None | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        if self._loaded_vars is None or skip or self.time >= self.horizon:
            return
        n_units = self.forecast_supply_days
        n_requests = self.forecast_demand_days
        pre_compute_horizon = self._loaded_vars_strides.shape[0]
        
        start_units = self.time - 1
        end_units = min(pre_compute_horizon - 1, self.time - 1 + n_units)
        slice_start_units = self._loaded_vars_strides[start_units]
        slice_end_units = self._loaded_vars_strides[end_units]
        units = self._loaded_vars['units'][slice_start_units[0]:slice_end_units[0], :]
        
        start_requests = self.time - 1
        end_requests = min(pre_compute_horizon - 1, self.time - 1 + n_requests)
        slice_start_requests = self._loaded_vars_strides[start_requests]
        slice_end_requests = self._loaded_vars_strides[end_requests]
        requests = self._loaded_vars['requests'][slice_start_requests[1]:slice_end_requests[1], :]
        requests_Abs = self._loaded_vars['requests_Abs'][slice_start_requests[1]:slice_end_requests[1], :]
        return units, (requests, requests_Abs)

    def _randomise_forecast(self, forecast: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]):
        if forecast is None:
            return forecast
        
        units, (requests, requests_Abs) = forecast
        u_shows = min(1, self.forecast_supply_shows)
        if u_shows < 1:
            raise NotImplementedError('Randomised forecasts with partial shows of units has not been updated yet.')
        else:
            show_units = units
        show_reqs = requests
        show_Abs = requests_Abs
        
        if not self._forecast_supply:
            show_units = np.empty((0, units.shape[1]), dtype=units.dtype)
        if not self._forecast_demand:
            show_reqs = np.empty((0, requests.shape[1]), dtype=requests.dtype)
            show_Abs = np.empty((0, requests_Abs.shape[1]), dtype=requests_Abs.dtype)
        return show_units, (show_reqs, show_Abs)


def simulate(sim: Simulation) -> Simulation:
    """Simulate a single replication of the simulation.
    
    Wrapper function for the `simulate` method of the `Simulation` class
    that allows for parallelisation.
    Inserts a 0.2 second pause before executing the simulation.
    
    :param Simulation sim: simulation to run
    return Simulation: the same simulation object
    """
    time.sleep(0.2)
    sim.simulate()
    return sim
