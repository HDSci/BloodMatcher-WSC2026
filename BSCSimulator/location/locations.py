import numpy as np
import pandas as pd

from ..antigen import Antigens
from ..util import UniqueIDTicker
from .demand import DEMAND_COLUMNS, LocationDemand
from .supply import SUPPLY_COLUMNS, LocationSupply
from .manufacturing import ManufacturingValidationRouting

SUPPLY = 'supply'
DEMAND = 'demand'
MANUFACTURING_ROUTING = 'manufacturing_routing'
REBALANCE_GROUP = 0
"""Pseudo-patient group for rebalancing stock across supply locations."""


class Locations:
    def __init__(self, location_data: dict, donor_data: dict, patient_data: dict,
                 antigens: Antigens, rngs: dict, seeds: dict = None,
                 courier_costs_df: pd.DataFrame = None,
                 transit_time_data_dfs: list[pd.DataFrame] = None):
        self.location_data = location_data  # {supply: {location_name: {supply_data}}, demand: {location_name: {demand_data}}}
        self.donor_data = donor_data
        self.patient_data = patient_data
        self.antigens = antigens
        self.locations_desc = {}  # {location_name: supply, demand, or transhipment}
        self.locations: dict[int, LocationSupply | LocationDemand | ManufacturingValidationRouting] = {}
        self.patient_groups = {}
        self.bulk_demand_groups = {}
        self.donor_groups = set()
        self.rngs = rngs
        self.seeds = seeds
        self.supply_locations_ids = []
        self.demand_locations_ids = []
        self.manufacturing_locations_ids = []
        self.transhipment_locations_ids = []
        self.population_abd_usabilities = []
        self.supply_locations_initial_inventories = []
        self.supply_locations_rebalance_points = []
        self._setup_locations()
        self.supply_locations_initial_inventories = np.array(self.supply_locations_initial_inventories)
        self.supply_locations_rebalance_points = np.array(self.supply_locations_rebalance_points)
        self.current_date = 0
        self._request_id_ticker = UniqueIDTicker()
        self._unit_id_ticker = UniqueIDTicker()
        self.rebalance_group_id = REBALANCE_GROUP
        self._setup_mappings()
        self._setup_courier_costs(courier_costs_df)
        self._setup_transit_time_data(transit_time_data_dfs)
        
    def _setup_locations(self):
        for loc_type, locs in self.location_data.items():
            for loc, loc_data in locs.items():
                self.locations_desc.update({loc: loc_type})
                if loc_type == SUPPLY:
                    self._setup_supply(loc, loc_data)
                elif loc_type == DEMAND:
                    self._setup_demand(loc, loc_data)
                elif loc_type == MANUFACTURING_ROUTING:
                    self._setup_manufacturing_routing(loc, loc_data)
        self.population_abd_usabilities = self._abod_usability()
                    
    def _setup_demand(self, location_name, location_data):
        self.patient_groups.update(
            {grp: grp_data['ID'] for grp, grp_data in location_data['PatientGroups'].items()})
        self.bulk_demand_groups.update(
            {grp: grp_data['ID'] for grp, grp_data in location_data['PatientGroups'].items() if grp_data['Type'] == 'Bulk'})
        id = location_data['UniqueID']
        rng_seed = self.rngs[DEMAND] if self.seeds is None else self.seeds[DEMAND]
        patient_data = self.patient_data[id]
        location_demand = LocationDemand(location_name, location_data, patient_data, rng_seed)
        self.locations.update({id: location_demand})
        self.demand_locations_ids.append(location_data['UniqueID'])        
        
    def _setup_supply(self, location_name, location_data):
        self.donor_groups.update(location_data['Supply']['DonorEthnicities'].keys())
        id = location_data['UniqueID']
        rng_seed = self.rngs[SUPPLY] if self.seeds is None else self.seeds[SUPPLY]
        donor_data = self.donor_data[id]
        location_supply = LocationSupply(location_name, location_data, donor_data, rng_seed)
        self.locations.update({id: location_supply})
        self.supply_locations_ids.append(location_data['UniqueID'])
        self.supply_locations_initial_inventories.append(location_supply.major_types_target_stock)
        self.supply_locations_rebalance_points.append(location_supply.major_types_rebalance_point)
        
    def _setup_manufacturing_routing(self, location_name, location_data):
        id = location_data['UniqueID']
        rng_seed = self.rngs[MANUFACTURING_ROUTING] if self.seeds is None else self.seeds[MANUFACTURING_ROUTING]
        manufacturing_routing = ManufacturingValidationRouting(location_name, location_data, rng_seed)
        self.locations.update({id: manufacturing_routing})
        self.manufacturing_locations_ids.append(location_data['UniqueID'])

    # TODO: Loop through locations to calculate the usability across the system.
    def _abod_usability(self):
        """Calculate and return the usability of ABD blood groups across the population.
        
        Right now, returns usability calculated on the 'old dummy demand'.
        :return ndarray: Usability of ABD blood groups across the population.
        """
        return np.array([1.0, 0.7872623036255959, 0.14842962857142858, 0.12103901126679187,
                         0.40050162857142857, 0.319065112870143, 0.030480942857142854, 0.023726709287880975])

    def tick(self):
        """Tick the locations forward by one day."""
        self.current_date += 1
        for loc in self.locations.values():
            loc.tick()

    def demand(self, rng: np.random.Generator = None, skip_demand=False):
        """Get demand for all locations.
        
        :param rng: Random number generator.
        :return tuple: Demand details and antibodies.
        """
        demand_detail = np.empty((0, DEMAND_COLUMNS), dtype=int)
        abs_masks = np.empty((0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        if skip_demand:
            return demand_detail, abs_masks
        for loc_id in self.demand_locations_ids:
            loc = self.locations[loc_id]
            demand_detail_loc, abs_masks_loc = loc.demand(self._request_id_ticker, rng)
            demand_detail = np.vstack((demand_detail, demand_detail_loc))
            abs_masks = np.vstack((abs_masks, abs_masks_loc))
        return demand_detail, abs_masks
        
    def supply(self, rng: np.random.Generator=None, initialise_inventory=False):
        """Get supply for all locations.
        
        :param Generator rng: Random number generator.
        :return ndarray: Supply details.
        """
        supply_detail = np.empty((0, SUPPLY_COLUMNS), dtype=int)
        for loc_id in self.supply_locations_ids:
            loc = self.locations[loc_id]
            supply_detail_loc = loc.supply(self._unit_id_ticker, initialise_inventory, rng)
            supply_detail = np.vstack((supply_detail, supply_detail_loc))
        return supply_detail
        
    def send_on_pathway(self, donations: np.ndarray, inventory, rng: np.random.Generator=None):
        """Send donations on the manufacturing/validation/routing pathway."""
        remaining_donations = None
        for loc_id in self.manufacturing_locations_ids:
            loc = self.locations[loc_id]
            remaining_donations = loc.start_pathway(remaining_donations if remaining_donations is not None else donations, inventory.deficit, rng)
 
    def receive_from_pathway(self,):
        """Receive products from the manufacturing/validation/routing pathway."""
        validated_available_units = np.empty((0, SUPPLY_COLUMNS), dtype=int)
        for loc_id in self.manufacturing_locations_ids:
            loc = self.locations[loc_id]
            validated_units = loc.finish_pathway()
            validated_available_units = np.vstack((validated_available_units, validated_units))
        return validated_available_units
 
    def initialise_routing(self, inventory):
        for loc_id in self.manufacturing_locations_ids:
            loc: ManufacturingValidationRouting = self.locations[loc_id]
            loc.initialise_routing(self.location_id_to_sink_id, inventory.loc_map, inventory.phen_map)
    
    def _setup_mappings(self):
        """Setup mappings for location IDs."""
        # Demand point to primary supply point
        if len(self.supply_locations_ids) == 1:
            # FIXME: Hacky solution to handle single supply location
            self.demand_to_primary_supply_map = np.full(max(self.demand_locations_ids) + 1, self.supply_locations_ids[0], dtype=int)
        else:
            # TODO: This would address the FIXME above - include the primary supply location in demand config file (and vice versa?)
            supply_points = {loc.name[4:]: loc.id for loc in self.locations.values() if self.locations_desc[loc.name] == SUPPLY and loc.name.startswith('SHU_')}
            demand_to_primary_supply = {loc.id: supply_points[loc.name[7:]] for loc in self.locations.values() if self.locations_desc[loc.name] == DEMAND and loc.name.startswith('Demand_')}
            demand_to_primary_supply_map = np.zeros(int(max(demand_to_primary_supply.keys())) + 1, dtype=int)
            demand_to_primary_supply_map[list(demand_to_primary_supply.keys())] = list(demand_to_primary_supply.values())
            self.demand_to_primary_supply_map = demand_to_primary_supply_map
        # Location unique ID to location sink ID
        location_ids = list(self.locations.keys())
        max_id = max(location_ids)
        location_id_to_sink_id = np.zeros(max_id + 1, dtype=int)
        location_id_to_sink_id[location_ids] = np.array([self.locations[loc_id].demand_id for loc_id in location_ids])
        self.location_id_to_sink_id = location_id_to_sink_id
        
    def _setup_courier_costs(self, costs_df: pd.DataFrame):
        """Setup courier costs matrix for all source-destination pairs."""
        location_ids = list(self.locations.keys())
        max_id = max(location_ids)
        max_sink_id = max(self.location_id_to_sink_id)
        cost_matrix = np.zeros((max_sink_id + 1, max_id + 1),)
        cost_matrix = np.zeros((max_id + 1, max_sink_id + 1))
        if costs_df is not None:
            df_rows = list(costs_df.index)
            df_cols = list(costs_df.columns)
            df_cols_int = np.array([int(col) for col in df_cols])
            for row in df_rows:
                cost_matrix[row, df_cols_int] = costs_df.loc[row, df_cols].to_numpy().flatten()
        self.courier_costs = cost_matrix.T

    def _setup_transit_time_data(self, transit_time_data: list[pd.DataFrame]):
        """Setup transit time data for all source-destination pairs for each day of the week."""
        location_ids = list(self.locations.keys())
        max_id = max(location_ids)
        max_sink_id = max(self.location_id_to_sink_id)
        transit_time_array = np.ones((7, max_id + 1, max_sink_id + 1), dtype=int)
        # Force zero transit time for pairs where the source is the same as the destination or is its primary supply point
        sink_ids_of_demand_locs = self.location_id_to_sink_id[self.demand_locations_ids]
        ids_of_primary_supply_locs = self.demand_to_primary_supply_map[self.demand_locations_ids]
        transit_time_array[:, ids_of_primary_supply_locs, sink_ids_of_demand_locs] = 0
        if transit_time_data is not None:
            for day, df in enumerate(transit_time_data):
                df_rows = list(df.index)
                df_cols = list(df.columns)
                df_cols_int = np.array([int(col) for col in df_cols])
                df_rows_int = np.array([int(row) for row in df_rows])
                for row in df_rows_int:
                    transit_time_array[day, row, df_cols_int] = df.loc[row, df_cols].to_numpy().flatten()
        self.transit_time_data = np.transpose(transit_time_array, (1, 2, 0))
