"""Location demand module."""

from collections.abc import ItemsView
from typing import Generator

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_discrete_frozen, rv_sample

from ..util import UniqueIDTicker

DEMAND_COLUMNS = 7
"""Columns -> 0: ID, 1: phenotype, 2: units, 3: due date, 4: patient group, 5: location ID, 6: location sink ID"""


class LocationDemand:
    """Location demand class."""

    def __init__(self, location_name: str, location_data: dict, patient_data: dict,
                 rng: int | np.random.Generator):
        """Initialize location demand."""
        self.name = location_name
        self.id: int = location_data['UniqueID']
        self.demand_id: int = location_data['DemandID']
        self.patient_groups: dict = location_data['PatientGroups']
        self.pat_group_names = list(self.patient_groups.keys())
        self.patient_data = patient_data
        self._setup_patient_data()
        self.rng = rng if not isinstance(rng, int) else np.random.default_rng(rng)
        self.current_date = 0
        
    def tick(self):
        """Tick the date forward."""
        self.current_date += 1
    
    # TODO: Add feature to read/use predetermined deterministic demand    
    def demand(self, id_ticker: UniqueIDTicker, rng: np.random.Generator = None):
        """Get demand for location."""
        rng = self.rng if rng is None else rng
        # init_tick = id_ticker + 0
        # Columns: ID, phenotype, units, due date, patient group, location ID, demand ID
        all_patients = np.empty((0, DEMAND_COLUMNS), dtype=int)
        all_abs_masks = np.empty((0, self._num_allotypes), dtype=bool)
        for grp, data in self.patient_groups.items():
            if data['Type'] == 'Appointments':
                # Generate appointments & patients
                patients = [[id_ticker.tick(True),
                             *self._appointment_demand(rng, data['UnitsperAppointment'], grp),
                             self.current_date, data['ID'],
                             self.id, self.demand_id] for _ in range(
                                self._generate_num_apps(data['Appointments'], rng))]
                # id_ticker += len(patients)
                # Generate patients' antibodies
                alloabs_mask = self._allo_antibodies(len(patients), grp, rng)
            elif data['Type'] == 'Bulk':
                # Generate bulk requests
                patients = [[id_ticker.tick(True), bg_type, units,
                             self.current_date, data['ID'],
                             self.id, self.demand_id] for _, (bg_type, units) in enumerate(
                                self._bulk_units(data['BulkUnits'], rng))]
                # id_ticker += len(patients)
                # Generate antibodies for bulk requests
                alloabs_mask = self._allo_antibodies(len(patients), grp, rng)
            if len(patients) == 0:
                continue
            all_patients = np.vstack((all_patients, patients))
            all_abs_masks = np.vstack((all_abs_masks, alloabs_mask))
        # assert init_tick + len(all_patients) == id_ticker + 0
        return all_patients, all_abs_masks

    def _appointment_demand(self, rng=None, units: int = None, patient_group: str = None):
        """Get demand for location."""
        rng = self.rng if rng is None else rng
        choices = self.patient_data[patient_group]['choices']
        probabilities = self.patient_data[patient_group]['probabilities']
        return np.array(rng.choice(choices, p=probabilities)), self._generate_num_units(units, rng)

    def _bulk_units(self, units: list[dict[int, rv_discrete_frozen | rv_sample]] | dict[int, int], rng: np.random.Generator = None) -> ItemsView[int, int] | Generator[tuple[int, int], None, None]:
        """Get bulk units for location."""
        rng = self.rng if rng is None else rng
        if isinstance(units, dict):
            return units.items()
        elif isinstance(units, list) and len(units) == 1:
            return ((bg, self._generate_num_units(u, rng)) for bg, u in units[0].items())
        elif isinstance(units, list) and len(units) > 1:
            day_of_week_units = units[self._get_day_of_week_index()]
            return ((bg, self._generate_num_units(u, rng)) for bg, u in day_of_week_units.items())
        raise ValueError('Invalid bulk units data structure.')

    def _generate_num_units(self, units, rng) -> int:
        """Generate number of units."""
        if isinstance(units, int):
            return units
        elif isinstance(units, tuple):
            raise NotImplementedError
            # return rng.randint(units[0], units[1])
        elif isinstance(units, (rv_discrete_frozen, rv_sample)):
            return units.rvs(random_state=rng)
        raise ValueError('Invalid units data structure.')

    def _generate_num_apps(self, appointments, rng) -> int:
        """Generate number of appointments."""
        if isinstance(appointments, int):
            return appointments
        elif isinstance(appointments, tuple):
            raise NotImplementedError
            # return rng.randint(appointments[0], appointments[1])
        elif isinstance(appointments, (rv_discrete_frozen, rv_sample)):
            return appointments.rvs(random_state=rng)
        elif isinstance(appointments, list):
            day_of_week_appointments: rv_discrete_frozen | rv_sample = appointments[self._get_day_of_week_index()]
            return day_of_week_appointments.rvs(random_state=rng)
        raise ValueError('Invalid appointments data structure.')

    def _allo_antibodies(self, num_patients, patient_group: str, rng: np.random.Generator = None, patients=None):
        """Generate antibodies."""
        freqs = self.alloab_data[patient_group]
        if np.all(freqs == 0):
            return np.zeros((num_patients, len(freqs)), dtype=bool)
        p = rng.uniform(size=(num_patients, len(freqs)))
        allo_ab_mask = p < freqs
        if patients is not None:
            # For using the patients phenotype to help generate antibodies
            # Either because we're using joint distributions of antibodies
            # Or to make sure the antibody mask reflects actual antibodies
            # I.e., no `True` values where the patient is positive for the antigen
            raise NotImplementedError
        return allo_ab_mask

    def _pad(self, blood_group_type, length):
        new_type = blood_group_type * (2 ** length)
        new_type += (2 ** length) - 1
        return new_type
    
    def _setup_patient_data(self):
        """Setup patient data and alloantibodies by converting it to numpy arrays."""
        self._patient_data = self.patient_data
        # self._allo_ab_data = self.patient_data['alloantibodies']
        patient_data = {}
        allo_ab_data = {}
        for grp, val in self._patient_data.items():
            if self.patient_groups[grp]['Type'] == 'Bulk':
                phens = list(val['phenotypes'].keys())
                probabilities = list(val['phenotypes'].values())
                patient_data.update(
                    {grp: {'choices': phens, 'probabilities': probabilities}})
            else:
                df_phens: pd.DataFrame = val['phenotypes']
                choices = df_phens.iloc[:, 0].to_numpy()
                probabilities = df_phens.iloc[:, 1].to_numpy()
                patient_data.update(
                    {grp: {'choices': choices, 'probabilities': probabilities}})
            df_abs: pd.DataFrame = val['alloantibodies']
            allo_ab_data.update({grp: df_abs.to_numpy().flatten()})
        self.patient_data = patient_data
        self.alloab_data = allo_ab_data
        self._num_allotypes = len(df_abs.columns)

    def _get_day_of_week_index(self,) -> int:
        """Get the index of the current day of the week in-simulation for data in lists."""
        return (self.current_date - 1) % 7
