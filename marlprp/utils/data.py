import numpy as np


def infer_num_storage_locations(num_skus, num_shelves, avg_loc_per_sku=None, num_storage_locations=None):
    size = (num_skus, num_shelves)
    if avg_loc_per_sku is not None and num_storage_locations is not None:
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



def schedule_sigmoid(current_epoch, max_epochs, beta=10):
    """
    Computes a smooth scheduling coefficient using a sigmoid function.
    
    Args:
        current_epoch (int): Current epoch number.
        max_epochs (int): Total number of training epochs.
        beta (float): Controls the sharpness of transition (higher means more abrupt).
    
    Returns:
        float: A value between 0 and 1, transitioning smoothly over epochs.
    """
    progress = current_epoch / max_epochs  # Normalize progress in [0,1]
    return 1 / (1 + np.exp(-beta * (progress - 0.5)))  # Sigmoid function


def environment_distribution(env_sizes, current_epoch, max_epochs, temperature=2.0, beta=10):
    """
    Compute a probability distribution over environments, shifting from size-based to uniform.

    Args:
        env_sizes (list or np.array): List of environment sizes.
        current_epoch (int): Current epoch number.
        max_epochs (int): Total number of training epochs.
        temperature (float): Controls initial preference for small environments.
        beta (float): Controls transition smoothness.

    Returns:
        np.array: Probability distribution over environments.
    """
    env_sizes = np.array(env_sizes)

    # Step 1: Compute initial probabilities favoring small environments
    initial_probs = np.exp(-temperature * env_sizes)
    initial_probs /= initial_probs.sum()  # Normalize to form a distribution

    # Step 2: Compute uniform distribution
    uniform_probs = np.ones_like(env_sizes) / len(env_sizes)

    # Step 3: Compute scheduling coefficient (alpha)
    alpha = schedule_sigmoid(current_epoch, max_epochs, beta)

    # Step 4: Interpolate between the two distributions
    final_probs = (1 - alpha) * initial_probs + alpha * uniform_probs

    # Ensure normalization (to account for numerical issues)
    final_probs /= final_probs.sum()

    return final_probs