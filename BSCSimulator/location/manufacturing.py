"""Location Manufacturing module."""

import numpy as np

from .supply import SUPPLY_COLUMNS


SHUS = 'SHUs'
ID = 'UniqueID'
DEM_ID = 'DemandID'
PAT_GRPS = 'PatientGroups'
PROCESSING_TIMES = 'ProcessingTimes'
PROCESSING_PATHS = 'SupplyPathways'
PATHS_ORDER = 'SupplyPathwaysOrder'


class ManufacturingValidationRouting:
    """Class that combines manufacturing, validation and routing."""
    
    def __init__(self, location_name: str, location_data: dict,
                 rng: int | np.random.Generator):
        """Initialize manufacturing, validation and routing
        """
        self.name = location_name
        self.id: int = location_data[ID]
        self.demand_id: int = location_data[DEM_ID]
        self.stock_holding_units: list[int] = location_data[SHUS]
        self.processing_paths: dict[str, dict] = location_data[PROCESSING_PATHS]
        self.processing_paths_order: list[str] = location_data[PATHS_ORDER]
        self.rng = rng if type(rng) is not int else np.random.default_rng(rng)
        self.loc_map: dict = None
        self.phen_map: dict = None
        self.location_id_to_sink_id: np.ndarray = None
        self.b_type_to_phenotype_name = {'R0': 'R0', 'Standard': 'TOTAL'}  # TODO: Remove hardcoding, must be a better way to do this
        self.pending_units = np.empty((0, SUPPLY_COLUMNS + 1), dtype=int)  # Columns: ID, phenotype, units, due date, location ID, location sink ID, date available
        self.current_date = 0
        
    def tick(self):
        """Tick the date forward."""
        self.current_date += 1
        
    def initialise_routing(self, location_id_to_sink_id: np.ndarray, loc_map: dict, phen_map: dict[str, int],):
        self.loc_map = loc_map
        self.phen_map = phen_map
        self.location_id_to_sink_id = location_id_to_sink_id
        
    def _select_units_in_region(self, donated_units: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
        """Select units in the region."""
        # shu_ids = list(self.stock_holding_units.keys())
        shu_ids = self.stock_holding_units
        selected_units_idx = np.isin(donated_units[:, 3], shu_ids)
        return donated_units[selected_units_idx], donated_units[~selected_units_idx]
    
    def start_pathway(self, donated_units: np.ndarray, inventory_deficit: np.ndarray,
                      rng: np.random.Generator = None, ):
        """Process/manufacture donations into RBC units then validate and route them.
        
        Pathway conditional on blood type of units.
        
        :param ndarray donated_units: Donated units pending processing and routing.
        :param ndarray inventory_deficit: Inventory deficit for each location on selected phenotypes.
        :param Generator rng: Random number generator
        """
        rng = self.rng if rng is None else rng
        # Select units that will go through this pathway
        region_units, non_region_units  = self._select_units_in_region(donated_units)
        _region_units = np.hstack((region_units.copy(), np.zeros((len(region_units), 1), dtype=int)))
        routed_units = self.pending_units[:0]
        for b_type_name in self.processing_paths_order:
            pathway_data = self.processing_paths[b_type_name]
            w_antigens = pathway_data['WatchedAntigens']
            w_phenotypes = pathway_data['WatchedPhenotype']
            paths: list[dict] = pathway_data['Paths']
            proportion_pathways = np.array(pathway_data['PathwaysProportionsCumulative'])
            b_type_units_idx = _region_units[:, 1] & w_antigens == w_phenotypes
            b_type_units = _region_units[b_type_units_idx]
            _b_type_units = np.split(b_type_units, np.round(
                proportion_pathways * len(b_type_units)).astype(int)[:-1], axis=0)
            for path_data, _btu in zip(paths, _b_type_units):
                dest_loc_ids: list = path_data['DestinationsIDs']
                dest_loc_dem_ids = self.location_id_to_sink_id[dest_loc_ids]
                dest_prop: np.ndarray = path_data['DestinationProportions']
                path_times: list = path_data['PathwayTimes']
                pheno_name = self.b_type_to_phenotype_name[b_type_name]
                len_btu = len(_btu) if len(_btu) > 0 else 1
                deficit_props = np.array(
                    [inventory_deficit[self.loc_map[loc_id], self.phen_map[pheno_name]] for loc_id in dest_loc_ids])/len_btu
                deficit_props[deficit_props < 0] = 0
                supply_props = dest_prop + deficit_props
                supply_props /= supply_props.sum()
                props_cumsum = np.cumsum(supply_props)
                end_idx = np.round(props_cumsum * len(_btu)).astype(int)
                path_units = np.split(_btu, end_idx[:-1], axis=0)
                for uid, usid, path_time, units in zip(dest_loc_ids, dest_loc_dem_ids, path_times, path_units):
                    units[:, 3:] = [uid, usid, self.current_date + path_time]
                    routed_units = np.vstack((routed_units, units))
            _region_units = _region_units[~b_type_units_idx]
        self.pending_units = np.vstack((self.pending_units, routed_units))
        return non_region_units

    def finish_pathway(self,) -> np.ndarray:
        """Finish processing/manufacturing, validating and routing."""
        finished_units_indices = self.pending_units[:, -1] <= self.current_date
        finished_units = self.pending_units[finished_units_indices]
        self.pending_units = self.pending_units[~finished_units_indices]
        return finished_units[:, :SUPPLY_COLUMNS]
