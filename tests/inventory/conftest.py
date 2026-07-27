import numpy as np
import pytest

from BSCSimulator.location.inventory import LocationInventory


@pytest.fixture
def shelf_life():
    return 23

@pytest.fixture
def inventory(shelf_life):
    return LocationInventory(shelf_life=shelf_life,
                             watched_antigens=np.array([114688, 114688, 114688, 114688, 114688, 114688, 114688, 114688, 31744, 128, 64, 0]),
                             watched_phenotypes=np.array([0, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 21504, 0, 0, 0]),
                             watched_phenotype_names=['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+', 'R0', 'Fya-', 'Fyb-', 'TOTAL'])

@pytest.fixture
def sample_units():
    # ID, antigen, date_bled, loc, sink, transit
    return np.array([
        [1, 0, 10, 1, 1, 0],  # age 0 at day 10
        [2, 0, 9, 1, 1, 0],   # age 1 at day 10
        [3, 0, 8, 1, 1, 0],   # age 2 at day 10
        [4, 0, 1, 1, 1, 0],   # age 22 at day 23, age 23 at day 24, expired starting day 25 if shelf life is 23
    ])

@pytest.fixture
def add_units_to_inventory(inventory, sample_units):
    inventory.current_date = 10
    inventory.add_to_store(sample_units[:,:-1])  # Exclude transit time column for adding to inventory
    return inventory
