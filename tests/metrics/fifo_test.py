import numpy as np

from BSCSimulator.metrics import fifo_discount, _fifo_discount


def test__fifo_discount():
    # Test that the FIFO discount is calculated correctly for a range of remaining shelf lives
    # shelf_life = 23
    remaining_shelf_lives = np.array([23, 12, 1, 0])
    expected_discounts = np.array([4.921566601e-03, 1/16, 0.793700526, 1.0])
    calculated_discounts = _fifo_discount(remaining_shelf_lives, decay_period=3,)
    # Test that the FIFO discount is calculated correctly for a range of remaining shelf lives
    assert np.allclose(calculated_discounts, expected_discounts)
    # Test that the FIFO discount is between 0 and 1 for non-expired units
    assert np.max(np.abs(calculated_discounts)) <= 1.0, "FIFO discounts should be between 0 and 1 for non-expired units"
    # Test that the FIFO discount is positive for non-expired units
    assert np.all(calculated_discounts > 0), "FIFO discounts should be positive for non-expired units"
    # Test that the FIFO discount is not out of bounds for remaining shelf lives greater than the shelf life
    remaining_shelf_lives = np.array([24, 25, 36, 44, 1_000])
    calculated_discounts = _fifo_discount(remaining_shelf_lives, decay_period=3,)
    assert np.all(calculated_discounts > 0.0) and np.all(calculated_discounts <= 1.0)


def test_fifo_discount():
    # Create a sample inventory with 4 units of blood with different ages
    # ID, antigen, date_bled, loc, sink, transit
    sample_inventory = np.array([
        [1, 0, 24, 1, 1, 0],  # age 0 at day 24
        [2, 0, 23, 1, 1, 0],  # age 1 at day 24
        [3, 0, 10, 1, 1, 0],  # age 14 at day 24
        [4, 0, 9, 1, 1, 0],   # age 15 at day 24
        [5, 0, 8, 1, 1, 0],   # age 16 at day 24
        [6, 0, 1, 1, 1, 0],   # age 23 at day 24 (due to expire the next day)
    ])
    current_date = 24
    shelf_life = 23
    # Calculate FIFO discount for each unit in the inventory
    discounts = fifo_discount(shelf_life + sample_inventory[:, 2] - current_date, max_life=shelf_life)
    # Check that the discounts are correct based on the ages of the units
    expected_discounts = np.array([4.921566601e-03, 6.200785359e-03, 0.125,
                                   0.1574901312, 0.1984251315, 1])
    assert np.allclose(discounts, expected_discounts[None, :])


def test_fifo_discount_with_future_units():
    # ID, antigen, date_bled, loc, sink, transit
    sample_inventory = np.array([
        [1, 0, 1, 1, 1, 0],   # age 23 at day 24 (due to expire the next day)
        [2, 0, 2, 1, 1, 0],   # age 22 at day 24
        [3, 0, 10, 1, 1, 0],   # age 14 at day 24
        [4, 0, 24, 1, 1, 0],   # age 0 at day 24
        [5, 0, 25, 1, 1, 0],   # Future date_bled (age -1 at day 24)
        [6, 0, 31, 1, 1, 0],   # Future date_bled (age -7 at day 24)
    ])
    current_date = 24
    shelf_life = 23
    # Calculate FIFO discount for each unit in the inventory
    discounts = fifo_discount(shelf_life + sample_inventory[:, 2] - current_date, max_life=shelf_life)
    # Check that the discounts are correct based on the ages of the units
    expected_discounts = np.array([1, 0.793700526, 0.125, 4.921566601e-03,
                                #    1/256, 1/1024])
                                      -1e16, -1e16]) # Future units have large negative discount to ensure they are not used for today's requests
    assert np.allclose(discounts, expected_discounts[None, :])


def test_fifo_discount_with_future_requests_units():
    # Create a sample inventory with 3 units of blood, where one unit has a future date_bled
    # ID, antigen, date_bled, loc, sink, transit
    sample_inventory = np.array([
        [1, 0, 1, 1, 1, 0],   # age 23 at day 24 (due to expire the next day)
        [4, 0, 24, 1, 1, 0],   # age 0 at day 24
        [6, 0, 31, 1, 1, 0],   # Future date_bled (age -7 at day 24)
    ])
    current_date = 24
    shelf_life = 23
    
    requests_dates = np.array([24, 25, 30, 31, 32])  # 1 request on the current day, the rest in the future

    # Calculate FIFO discount for each unit × request
    discounts = fifo_discount(shelf_life + sample_inventory[:, 2]- current_date, shelf_life,
                              reqs_dates=requests_dates - current_date)

    # Check that the discounts are correct based on the ages of the units and the request dates
    expected_discounts = np.array([[1, 4.921566601e-03, -1e16],  # For request on day 24: unit 1 has discount 1, unit 4 has discount ~0.0049, unit 6 allocating to past request has large negative discount
                                   # For request on day 25: unit 1 has expired, unit 4 has discount ~0.0049, unit 6 allocating to the past
                                   [-1e16, 6.200785359e-03, -1e16],
                                   # For request on day 30: unit 1 has expired, unit 4 has discount ~0.0197, unit 6 allocating to the past
                                   [-1e16, 0.0196862664, -1e16],
                                   # For request on day 31: unit 1 has expired, unit 4 has discount ~0.0248, unit 6 has discount ~0.0049
                                   [-1e16, 0.0248031414, 4.921566601e-03],
                                   # For request on day 32: unit 1 has expired, unit 4 has discount 1/32, unit 6 has discount ~0.0062
                                   [-1e16, 1/32, 6.200785359e-03]
                                   ])
    assert discounts.shape == (5, 3), "Discounts array should have shape (num_requests, num_units)"
    assert np.allclose(discounts, expected_discounts)
