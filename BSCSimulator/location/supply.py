"""Location supply module."""

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_discrete_frozen, rv_sample

from ..util import UniqueIDTicker

SUPPLY_COLUMNS = 5
"""Columns -> 0: ID, 1: antigen vector, 2: date bled, 3: location ID, 4: location sink ID"""

REBALANCE_POINT = 6
"""Number of days of stock to trigger rebalance"""

class LocationSupply:
    """Location supply class."""

    def __init__(self, location_name: str, location_data: dict, donor_data: pd.DataFrame,
                 rng: int | np.random.Generator):
        """Initialize location supply."""
        self.name = location_name
        self.id = location_data['UniqueID']
        self.demand_id = location_data['DemandID']
        self.supply_data: dict = location_data['Supply']
        self.donor_groups = self.supply_data['DonorEthnicities']
        self.donor_group_names = list(self.donor_groups.keys())
        self.donor_data = donor_data
        self._setup_donor_data()
        self.rng = rng if not isinstance(rng, int) else np.random.default_rng(rng)
        self.current_date = 0
        self.inventory_thresholds = None
        self.order_point = self.supply_data['OrderPoint']
        self._measure_major_types_initial_inventory()

    def tick(self):
        """Tick the date forward."""
        self.current_date += 1
        
    def supply(self, id_ticker: UniqueIDTicker, initialise_inventory=False, rng=None):
        """Get supply for location."""
        rng = self.rng if rng is None else rng
        all_donations = np.empty((0, SUPPLY_COLUMNS), dtype=int)
        # Generate donations
        if initialise_inventory:
            num_donations = self.supply_data['InitialInventory']
        else:
            num_donations = self._generate_num_donations(self.supply_data['Donations'], rng)
        phenos = self._supply(rng, num_donations)[:, None]
        ids = np.arange(id_ticker.get(), id_ticker + num_donations)[:, None]
        dates = np.full(num_donations, self.current_date, dtype=int)[:, None]
        u_ids = np.full(num_donations, self.id, dtype=int)[:, None]
        d_ids = np.full(num_donations, self.demand_id, dtype=int)[:, None]
        donations = np.hstack((ids, phenos, dates, u_ids, d_ids))
        id_ticker.increment(num_donations)
        # id_ticker += num_donations
        all_donations = np.vstack((all_donations, donations))
        return all_donations
    
    def _supply(self, rng, units=1):
        """Generate supply."""
        choices = self.donor_data['choices']
        probabilities = self.donor_data['probabilities']
        result = rng.choice(choices, p=probabilities, size=units)
        return np.array(result)

    def _generate_num_donations(self, donations: int | tuple | rv_discrete_frozen | rv_sample | list[rv_discrete_frozen], rng):
        """Generate number of donations."""
        if isinstance(donations, int):
            return donations
        elif isinstance(donations, tuple):
            raise NotImplementedError
            # return rng.integers(donations[0], donations[1])
        elif isinstance(donations, (rv_discrete_frozen, rv_sample)):
            return donations.rvs(random_state=rng)
        elif isinstance(donations, list):
            day_of_week_donations = donations[self._get_day_of_week_index()]
            return day_of_week_donations.rvs(random_state=rng)
        raise ValueError('Invalid donations data structure.')
        
    def _setup_donor_data(self):
        """Setup donor data by converting datframes to numpy arrays."""
        self._donor_data = self.donor_data
        donor_data = {}
        # for grp, df in self._donor_data.items():
        choices = self.donor_data.iloc[:, 0].to_numpy(copy=True)
        probabilities = self.donor_data.iloc[:, 1].to_numpy(copy=True)
        probabilities /= probabilities.sum()
        donor_data.update({'choices': choices, 'probabilities': probabilities})
        self.donor_data = donor_data
    
    def _measure_major_types_initial_inventory(self):
        """Measure major types in initial inventory."""
        choices = self.donor_data['choices']
        probabilities = self.donor_data['probabilities']
        abod_choices = choices >> 14
        abod_probabilities = []
        for i in range(8):
            abod_probabilities.append(probabilities[(abod_choices == i)].sum())
        abod_probabilities = np.array(abod_probabilities)
        self.major_types_target_fraction = abod_probabilities / abod_probabilities.sum()
        self.major_types_target_stock = np.floor(
            self.supply_data['InitialInventory'] * self.major_types_target_fraction).astype(int)
        self.major_types_rebalance_point = np.ceil(self.order_point * self.major_types_target_fraction).astype(int)

    def _get_day_of_week_index(self,) -> int:
        """Get the index of the current day of the week in-simulation for data in lists."""
        return (self.current_date - 1) % 7
