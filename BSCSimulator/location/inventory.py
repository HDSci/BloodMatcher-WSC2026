"""Location Inventory module."""

from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .locations import Locations

DEFAULT_SHELF_LIFE = 35

INVENTORY_COLUMNS = 6
"""Columns-> 0: ID, 1: antigen vector, 2: date bled, 3: location id, 4: sink id, 5: remaining time in transit"""

MOVES_COLUMNS = INVENTORY_COLUMNS + 4
"""Columns-> 0: ID, 1: antigen vector, 2: date bled, 3: location id, 4: sink id, 5: transit time,
6: date moved, 7: request due date, 8: patient_group, 9: new location id"""

MAJOR_BLOOD_TYPES = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']

class LocationInventory:

    def __init__(
            self, shelf_life: int = DEFAULT_SHELF_LIFE, inventory_size: int = 30_000,
            watched_antigens: np.ndarray = None,
            watched_phenotypes: np.ndarray = None, watched_phenotype_names: np.ndarray = None,) -> None:
        """Initialise the inventory.
        
        Inventory of blood units.
        
        :param int shelf_life: The shelf life of the units in the inventory, in days.
        :param ndarray watched_antigens: 1-D integer array of
        masks selecting the antigens in phenotype combinations to watch
        :param ndarray watched_phenotypes: 1-D integer array of
        phenotype combinations to watch
        :param ndarray watched_phenotype_names: 1-D string array of
        names of the watched phenotype combinations
        """
        self.shelf_life = shelf_life
        # Columns: ID, antigen vector, date bled, location id, demand id
        self.store = np.empty(shape=(0, INVENTORY_COLUMNS), dtype=int)
        """Columns: 0: ID, 1: antigen vector, 2: date bled, 3: location id, 4: sink id, 5: remaining time in transit"""
        self.expired = np.empty(shape=(0, INVENTORY_COLUMNS), dtype=int)
        """Columns: 0: ID, 1: antigen vector, 2: date bled, 3: location id, 4: sink id, 5: remaining time in transit"""
        self._start_date = 0
        self.current_date = self._start_date
        self.inventory_size = inventory_size
        self.stock_levels = []
        self.stock_levels_locations_raw = []
        self.available_stock_levels = []
        self.watched_antigens = watched_antigens
        self.watched_phenotypes = watched_phenotypes
        self.watched_phenotype_names = watched_phenotype_names
        self.age_distribution = np.empty(shape=(0, self.shelf_life+1), dtype=int)
        self.age_distribution_locs = []
        self._contains_phenotype_combo = None
        self._contains_phenotype_combo_at_location = None
        self.pheno_age_dist = [ [] for _ in range(len(self.watched_phenotypes))]
        self.pheno_age_dist_loc = None
        # Columns: ID, antigen vector, date bled, old location id, demand id, date moved, new location id
        self.recorded_moves = np.empty(shape=(0, MOVES_COLUMNS), dtype=int)
        """Columns-> 0: ID, 1: antigen vector, 2: date bled, 3: location id, 4: demand id, 5: transit time,
        6: date moved, 7: request due date, 8: patient_group, 9: new location id"""
        self.major_types_indices = []
        for x in MAJOR_BLOOD_TYPES:
            self.major_types_indices.append(np.where(
                np.atleast_1d(watched_phenotype_names) == np.atleast_1d(x))[0][0])
        self.major_types_indices = np.array(self.major_types_indices)
        self.major_types_stock_at_locations = np.empty(shape=(0, 8), dtype=int)
        self.deficit = np.empty(shape=(0, len(watched_phenotypes)), dtype=int)
        self.supply_locations_thresholds = np.empty(shape=(0, len(watched_phenotypes)), dtype=int)

    def tick(self):
        """Increment the current date by one day."""
        self.current_date += 1

    def add_to_store(self, units: ArrayLike):
        """Add units to the inventory.
        
        :param ndarray units: rows of units to add to the inventory.
        Columns: ID, antigen vector, date bled, location id, demand id
        """
        _units = np.atleast_2d(units)
        if _units.size <= 0:
            return
        _units = np.hstack((_units, np.zeros((_units.shape[0], 1), dtype=int)))
        self.store = np.vstack((self.store, _units))

    def remove_from_store(self, units: ArrayLike):
        """Remove units from the inventory.
        
        :param ndarray units: rows of units in the inventory to remove
        """
        units = np.atleast_2d(units)
        _not_i = np.isin(self.store[:, 0], units[:, 0], assume_unique=True, invert=True)
        self.store = self.store[_not_i, :]

    def units_younger_than(self, age: int):
        """Return units younger than a given age.
        
        :param int age: age in days
        :return ndarray: units younger than age
        """
        i = self.store[:, 2] > self.current_date - age
        return self.store[i, :]

    def units_older_than(self, age: int):
        """Return units older than a given age.
        
        :param int age: age in days
        :return ndarray: units older than age
        """
        i = self.store[:, 2] < self.current_date - age
        return self.store[i, :]
    
    def units_from_location(self, location_id: int):
        """Return units from a given location.
        
        :param int location_id: location id
        :return ndarray: units from location
        """
        i = self.store[:, 3] == location_id
        return self.store[i, :]
    
    def units_from_locations(self, location_ids: np.ndarray):
        """Return units from given locations.
        
        :param ndarray location_ids: 1-D integer array of location ids
        :return ndarray: units from locations
        """
        i = np.isin(self.store[:, 3], location_ids)
        return self.store[i, :]
    
    def units_with_remaining_transit_time_of(self, days: int):
        """Return units with remaining transit time of given days.
        
        When `days` is 0, this returns all units that are not in transit.
        
        :param int days: time in transit (measured in days) until the unit(s) arrive(s)
        :return ndarray: units with remaining transit time of `days`
        """
        i = self.store[:, 5] == days
        return self.store[i, :]
    
    def units_with_remaining_transit_time_less_than(self, days: int, inclusive: bool = False):
        """Return units with remaining transit time less than given days.
        
        :param int days: time in transit (measured in days) until the unit(s) arrive(s)
        :param bool inclusive: whether to include units with remaining transit time equal to `days`
        :return ndarray: units with remaining transit time less than `days`
        """
        if inclusive:
            i = self.store[:, 5] <= days
        else:
            i = self.store[:, 5] < days
        return self.store[i, :]
    
    def units_with_remaining_transit_time_greater_than(self, days: int, inclusive: bool = False):
        """Return units with remaining transit time greater than given days.
        
        :param int days: time in transit (measured in days) until the unit(s) arrive(s)
        :param bool inclusive: whether to include units with remaining transit time equal to `days`
        :return ndarray: units with remaining transit time greater than `days`
        """
        if inclusive:
            i = self.store[:, 5] >= days
        else:
            i = self.store[:, 5] > days
        return self.store[i, :]

    def remove_expired_units(self):
        """Remove expired units from the inventory.
        
        Assumes to be called at the end of the day.
        So, units at the end of their shelf life are removed
        and recorded in the expired units - `self.expired`.
        """
        expired = self.units_older_than(self.shelf_life - 1)
        if expired.size == 0:
            return
        self.expired = np.vstack((self.expired, expired))
        self.store = self.units_younger_than(self.shelf_life)

    #TODO: Add functionality to determine or save the inventory thresholds for each location
    def initialise_inventory(self, locations: Locations):
        """Initialise the inventory with units from the supply.
        
        :param Locations locations: Locations object
        """
        self.loc_map = {sid: i for i, sid in enumerate(locations.supply_locations_ids)}
        self.phen_map = {name: i for i, name in enumerate(self.watched_phenotype_names)}
        self.supply_locations_thresholds = np.zeros((len(locations.supply_locations_ids), len(self.watched_phenotypes)), dtype=int)
        for loc_id in locations.supply_locations_ids:
            loc_idx = self.loc_map[loc_id]
            thresholds = locations.locations[loc_id].major_types_rebalance_point
            self.supply_locations_thresholds[loc_idx, :len(thresholds)] = thresholds
            # TODO: Hacky, fix this
            self.supply_locations_thresholds[loc_idx, -1] = np.sum(thresholds)
        self.deficit = np.zeros((len(locations.supply_locations_ids), len(self.watched_phenotypes)), dtype=int)
        
    def warmup_clear(self):
        """Clear the inventory records after the warmup period."""
        self.expired = np.empty(shape=(0, INVENTORY_COLUMNS), dtype=int)
        self.recorded_moves = np.empty(shape=(0, MOVES_COLUMNS), dtype=int)

    def move_units(self, units_ids: np.ndarray, location_ids: np.ndarray, request_due_dates: np.ndarray,
                   patient_groups: np.ndarray, location_sink_ids: np.ndarray, transit_times: np.ndarray,
                   record: bool = False):
        """Move units to given locations.
        
        :param ndarray units:  1-D integer array of ids for units to move
        :param ndarray location_ids: 1-D integer array of destination location ids
        :param ndarray request_due_dates: 1-D integer array of due dates for the requests
        :param bool record: whether to record the moves
        """
        # Start moving units
        if units_ids.size == 0 or location_ids.size == 0:
            # Continue transit of other units
            self._continue_transit()
            return
        is_in_store = np.isin(units_ids, self.store[:, 0])
        if not is_in_store.all():
            raise ValueError("Some units are not in the inventory.")
        i = np.isin(self.store[:, 0], units_ids)
        old_ids = self.store[i, 3:5]
        units_id_in_order_of_store = self.store[i, 0]
        # Vectorized mapping: find indices of units_id_in_order_of_store in units_ids  
        sorter = np.argsort(units_ids)  
        positions_of_ids = sorter[np.searchsorted(units_ids, units_id_in_order_of_store, sorter=sorter)] 
        # id_to_index = {uid: idx for idx, uid in enumerate(units_ids)}
        # positions_of_ids = np.array([id_to_index[uid] for uid in units_id_in_order_of_store])
        reordered_location_ids = location_ids[positions_of_ids]
        reordered_location_sink_ids = location_sink_ids[positions_of_ids]
        self.store[i, 3] = reordered_location_ids
        self.store[i, 4] = reordered_location_sink_ids
        self.store[i, 5] = transit_times[positions_of_ids]
        # Record the moves
        if record:
            self._record_moves(old_ids, request_due_dates[positions_of_ids],
                               patient_groups[positions_of_ids], i)
        # Continue transit of other units
        self._continue_transit()

    def _record_moves(self, old_location_ids: np.ndarray, request_due_dates: np.ndarray,
                      patient_groups: np.ndarray, i: np.ndarray):
        """Record the moves of units to locations.
        
        :param ndarray old_location_ids: 2-D integer array of old location IDs & sink IDs
        :param ndarray request_due_dates: 1-D integer array of due dates for the requests pulling the units
        :param ndarray patient_groups: 1-D integer array of patient groups for the requests
        :param ndarray i: boolean array selecting units to move
        """
        units_in_store = self.store[i, :]
        times = np.full((units_in_store.shape[0], 1), self.current_date, dtype=int)
        moves_to_record = np.hstack(
            (units_in_store[:, :3], old_location_ids, units_in_store[:, 5:6],
             times, request_due_dates[:, None], patient_groups[:, None], units_in_store[:, 3:4]))
        self.recorded_moves = np.vstack((self.recorded_moves, moves_to_record))

    def _continue_transit(self) -> None:
        """Decrement the remaining transit time of units in transit by one day."""
        in_transit = self.store[:, 5] > 0
        self.store[in_transit, 5] -= 1

    def request_rebalance_of_stock(self, locations: Locations, rebalance_lead_time = 1) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Determines if stock levels are low and requests rebalance.
        
        Determines if the stock levels of the major blood types are low
        in the inventory of each SHU location and calculates the number of units
        needed to rebalance the stock levels compared to the initial inventory.
        
        :param Locations locations: Locations object with list of SHU locations
        :param int rebalance_lead_time: lead time in days to allow the units to arrive
        :return tuple: rebalance order, dummy Abs, sources
        """
        need_rebalance = self.major_types_stock_at_locations < locations.supply_locations_rebalance_points
        if not np.any(need_rebalance):
            return None
        need_rebalance_locations = np.any(need_rebalance, axis=1)
        amounts_to_request = np.zeros(need_rebalance.shape, dtype=int)
        raw_deviations = locations.supply_locations_initial_inventories - self.major_types_stock_at_locations
        amounts_to_request[need_rebalance] = raw_deviations[need_rebalance]
        
        loc, major_type = np.where(need_rebalance)
        ids = np.arange(-1, -need_rebalance.size-1, -1)[need_rebalance.flatten()]
        phenotypes = (major_type << 14) + (2 ** 14 - 1)  # TODO: Fix this hacky way of getting phenotypes. Use the antigen vector length to determine how many bits to shift
        units = amounts_to_request[need_rebalance]
        # TODO: I think that maybe the rebalance request need to have a due date a few days in the future. So rebalance request made on Thursday for arrival on Monday...?
        dates = np.full(units.size, self.current_date + rebalance_lead_time, dtype=int)  # Date + 1 incidentally avoids using expiring units
        patient_group = np.full(units.size, locations.rebalance_group_id, dtype=int)
        location_ids = np.array(locations.supply_locations_ids)[loc]
        demand_ids = np.array([locations.locations[l].demand_id for l in location_ids])
        rebalance_order = np.hstack((ids[:, None], phenotypes[:, None], units[:, None],
                                     dates[:, None], patient_group[:, None], 
                                     location_ids[:, None], demand_ids[:, None]), dtype=int)
        dummy_Abs = np.zeros((rebalance_order.shape[0], len(locations.antigens.alloantibody_freqs)), dtype=bool)
        
        can_be_sources = self.major_types_stock_at_locations > locations.supply_locations_initial_inventories
        sources = [np.array(locations.supply_locations_ids)[can_be_sources[:, mt]] for mt in major_type]
        return rebalance_order, dummy_Abs, sources
    
    def measure_stock(self, locations):
        """Measure the stock levels and age distribution of the inventory."""
        self.measure_stock_levels(locations)
        self.major_types_stock_at_locations = self.stock_levels_locations_raw[-1][:, self.major_types_indices]
        if len(self.stock_levels_locations_raw) <= 1:
            stock_levels_at_locations = 0
        else:
            stock_levels_at_locations = self.stock_levels_locations_raw[-1]
        self.deficit = self.supply_locations_thresholds - stock_levels_at_locations
        self.measure_age_distribution(locations)
    
    def measure_stock_levels(self, locations: Locations, supply: np.ndarray = None, not_in_transit: np.ndarray = None,
                             watched_antigens: np.ndarray = None, watched_phenotypes: np.ndarray = None) -> None:
        """Measure stock levels of watched phenotypes.
        
        Measures and then stores the stock levels for strategic antigen combinations.
        These stock levels are measured as a percentage of the total stock of the supply.
        
        :param Locations locations: Locations object from which to get location ids
        :param ndarray supply: 1-D integer array of antigen phenotypes of the supply
        :param ndarray not_in_transit: 1-D boolean array of whether each unit is not in transit
        :param ndarray wathced_antigen: 1-D integer array of masks selecting the antigens in phenotype combinations to watch
        :param ndarray watched_phenotypes: 1-D integer array of phenotype combinations to watch
        """
        if watched_antigens is None:
            watched_antigens = self.watched_antigens
        if watched_phenotypes is None:
            watched_phenotypes = self.watched_phenotypes
        store = self.store
        location_ids = np.array(locations.supply_locations_ids)
        if supply is None:
            supply = self.store[:, 1]
            not_in_transit = self.store[:, 5] <= 0
        if not_in_transit is None:
            not_in_transit = np.ones(supply.shape, dtype=bool)
        contains_phenotype_combo = supply & watched_antigens[:, None] == watched_phenotypes[:, None]  # shape: (phenotypes, units)
        unit_locations = store[:, 3][:, None] == location_ids[:, None, None]  # shape: (locations, units, 1)
        contains_phenotype_combo_at_location = np.swapaxes(contains_phenotype_combo[:, :, None], 0, 2) & unit_locations
        stock_levels_at_location = contains_phenotype_combo_at_location.sum(axis=1)  # shape: (locations, phenotypes)
        stock_levels = contains_phenotype_combo.sum(axis=1)
        available_stock_levels_at_locations = contains_phenotype_combo_at_location[:, not_in_transit, :].sum(axis=(1, 2))
        total_stock = len(supply)
        stock_levels = stock_levels / (total_stock if total_stock > 0 else 1)
        self.stock_levels_locations_raw.append(stock_levels_at_location)
        self.stock_levels.append(np.append(stock_levels, total_stock))
        self.available_stock_levels.append(available_stock_levels_at_locations)
        self._contains_phenotype_combo = contains_phenotype_combo
        self._contains_phenotype_combo_at_location = contains_phenotype_combo_at_location
    
    def measure_age_distribution(self, locations: Locations, supply: np.ndarray = None, shelf_life: int = None,
                                 for_phenotypes=True) -> None:
        """Measure the age distribution of the inventory.
        
        Measures and then stores the age distribution of the inventory.
        The age distribution is measured as the number of units of each age.
        
        :param ndarray supply: 1-D integer array of ages of stored units
        :param int shelf_life: maximum shelf life of units
        """
        if supply is None:
            supply = self.current_date - self.store[:, 2]
        if shelf_life is None:
            shelf_life = self.shelf_life
        age_distribution = np.bincount(supply, minlength=shelf_life+1)
        self.age_distribution = np.vstack((self.age_distribution, age_distribution))
        location_ids = np.array(locations.supply_locations_ids)
        age_dist_locs = [np.bincount(supply[self.store[:, 3] == loc], minlength=shelf_life+1) for loc in location_ids]
        self.age_distribution_locs.append(np.array(age_dist_locs)) 
        
        if self.pheno_age_dist_loc is None:
            self.pheno_age_dist_loc = [[[] for _j in self.watched_phenotypes] for _i in location_ids]
        
        if for_phenotypes and self._contains_phenotype_combo is not None:
            contains_phenotype_combo = self._contains_phenotype_combo
            pheno_age_dist = [np.bincount(supply[combo], minlength=shelf_life+1) for combo in contains_phenotype_combo]
            for combo, age_dist in zip(self.pheno_age_dist, pheno_age_dist):
                combo.append(age_dist)
        if for_phenotypes and self._contains_phenotype_combo_at_location is not None:
            contains_phenotype_combo_at_location = np.swapaxes(self._contains_phenotype_combo_at_location, 1, 2)  # shape: (locations, phenotypes, units)
            pheno_age_dist_loc = [[np.bincount(supply[combo], minlength=shelf_life+1)for combo in loc]
                                  for loc in contains_phenotype_combo_at_location]
            for loc, loc_age_dist in zip(self.pheno_age_dist_loc, pheno_age_dist_loc):
                for combo, age_dist in zip(loc, loc_age_dist):
                    combo.append(age_dist)
                # combo.append(age_dist)
        self._contains_phenotype_combo = None
        self._contains_phenotype_combo_at_location = None
    
    def mean_O_type_stock(self, start: int = 1, end: int = np.inf) -> np.ndarray:
        """Return the mean stock of O type blood units.
        
        Measures the mean levels of both O- and O+ blood units
        as a percentage of the total stock of the supply
        over the specified period in the simulation.
        
        Assumes that the first two watched phenotypes are O- and O+.
        
        :param int start: start date of period
        :param int end: end date of period
        :return ndarray: mean stock of O type blood units
        """
        start_index = max(start - 1, 0)
        end_index = min(end, len(self.stock_levels))
        stock_levels = np.array(self.stock_levels)[start_index:end_index, :2]
        mean_stock_levels = stock_levels.mean(axis=0)
        if mean_stock_levels.size == 0:
            mean_stock_levels = np.zeros(2)
        elif mean_stock_levels.size == 1:
            mean_stock_levels = np.append(mean_stock_levels, 0)
        return mean_stock_levels    
