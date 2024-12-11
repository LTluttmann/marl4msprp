import numpy as np
from rl4co.utils import pylogger


log = pylogger.get_pylogger(__name__)


def infer_num_storage_locations(num_skus, num_shelves, avg_loc_per_sku=None, num_storage_locations=None):
    size = (num_skus, num_shelves)
    if avg_loc_per_sku is not None and num_storage_locations is not None:
        log.info("Warning! Set both, avg_loc_per_sku and num_storage_locations. I will ignore avg_loc_per_sku")
        assert num_storage_locations >= num_shelves
        assert num_storage_locations <= np.prod(size)
        num_storage_locations = num_storage_locations
    elif avg_loc_per_sku is None and num_storage_locations is None:
        raise ValueError("Specify either avg_loc_per_sku or num physical items")
    elif avg_loc_per_sku is not None:
        assert avg_loc_per_sku < num_shelves
        num_storage_locations = max(num_shelves, int(num_skus * avg_loc_per_sku))
    else: 
        assert num_storage_locations >= num_shelves
        assert num_storage_locations <= np.prod(size)
        num_storage_locations = num_storage_locations
    return num_storage_locations
