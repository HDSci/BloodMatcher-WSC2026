import numpy as np


def test_units_younger_than(add_units_to_inventory):
    inventory = add_units_to_inventory
    younger_than_2 = inventory.units_younger_than(2)
    # Basic dimension check
    assert younger_than_2.shape[1] == 6
    # Check that only units with age < 2 are included
    assert set(younger_than_2[:, 0]) == {1, 2}  # Check that only units 1 and 2 are included
    # Check that ages are correct
    ages = inventory.current_date - younger_than_2[:, 2]
    assert np.all(ages < 2)
    

def test_units_older_than(add_units_to_inventory):
    inventory = add_units_to_inventory
    older_than_1 = inventory.units_older_than(1)
    # Basic dimension check
    assert older_than_1.shape[1] == 6
    # Check that only units with age > 1 are included
    assert set(older_than_1[:, 0]) == {3, 4}
    # Check that ages are correct
    ages = inventory.current_date - older_than_1[:, 2]
    assert np.all(ages > 1)

def test_remove_expired_units(add_units_to_inventory):
    inventory = add_units_to_inventory
    # Test that no units are removed when current date is before any expiration
    inventory.remove_expired_units()
    assert inventory.store.shape[0] == 4
    assert inventory.expired.size == 0
    # Advance current date to 24, which should expire unit 4
    inventory.current_date = 24
    # Test that at the end of day 24, unit 4 will be removed
    inventory.remove_expired_units()
    assert inventory.store.shape[0] == 3
    assert inventory.expired.shape[0] == 1
    assert inventory.expired[0, 0] == 4  # Check that the expired unit is unit 4
    # Advance current date to 25, which should not remove any additional units since unit 4 is already removed
    inventory.current_date = 25
    inventory.remove_expired_units()
    assert inventory.store.shape[0] == 3
    assert inventory.expired.shape[0] == 1  # No new expired units
    assert inventory.expired[0, 0] == 4  # Still only unit 4 is expired
    # Advance current date to 32, which should expire unit 2 and unit 3 (which should have expired the day before)
    inventory.current_date = 32
    inventory.remove_expired_units()
    assert inventory.store.shape[0] == 1  # Only unit 1 should remain
    assert inventory.expired.shape[0] == 3  # Units 2 and 3 should now be expired
    assert set(inventory.expired[:, 0]) == {2, 3, 4}  # Check that the expired units are unit 2, 3, and 4. Tests that any missed expirations are caught in subsequent calls to remove_expired_units()
