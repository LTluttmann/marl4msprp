import torch
import numpy as np
import logging
from tensordict import TensorDict

from marlprp.utils.config import EnvParams


log = logging.getLogger(__name__)


class MSPRPGenerator:

    def __init__(
        self,
        instance_params: EnvParams
    ):

        super(MSPRPGenerator, self).__init__()

        np.random.seed(instance_params.seed)
        torch.manual_seed(instance_params.seed)

        self.num_depots = instance_params.num_depots
        self.num_skus = instance_params.num_skus
        self.num_shelves = instance_params.num_shelves
        self.size_tuple = (self.num_shelves, self.num_skus)

        self.num_storage_locations = instance_params.num_storage_locations
        log.info(f"Instance size is {self.num_storage_locations} physical items")

        self.capacity = instance_params.capacity
        log.info(f"Picker capacity is {self.capacity}")


        # get demand params
        self.min_demand = instance_params.min_demand
        self.max_demand = instance_params.max_demand

        # calc supply params
        # -------- old -----------
        # self.lamda = instance_params.lam
        # self.min_supply = instance_params.min_supply
        # ------------------------
        
        self.min_supply = instance_params.min_supply
        if instance_params.max_supply is None:
            avg_demand_per_sku = (self.max_demand + self.min_demand) / 2
            avg_total_supply_per_sku = instance_params.avg_supply_to_demand_ratio * avg_demand_per_sku
            avg_storgage_locations_per_sku = self.num_storage_locations / self.num_skus
            avg_supply_per_storage_loc = max(avg_total_supply_per_sku / avg_storgage_locations_per_sku, 1)
            self.max_supply = np.ceil(avg_supply_per_storage_loc * 2 - self.min_supply).astype("int")
        else:
            self.max_supply = instance_params.max_supply

        self.num_agents = instance_params.num_agents

    def _simulate_batch(self, batch_size: tuple):
        # simulate supply [BS, P, S]

        supply = torch.randint(
            low=self.min_supply, 
            high=self.max_supply+1, 
            size=(*batch_size, self.num_shelves, self.num_skus),
            dtype=torch.float32
        )

        # simulate demand [BS, P]; add 1 since max of randint is 'high'-1
        demand = torch.randint(
            low=self.min_demand, 
            high=self.max_demand+1, 
            size=(*batch_size, self.num_skus),
            dtype=torch.float32
        )

        num_nodes = self.num_shelves + self.num_depots
        # simulate shelf locations as x,y coordinates in a unit circle [BS, S, 2]
        coordinates = torch.rand((*batch_size, num_nodes, 2))

        # simulate for each batch a series of indices which correspond to the item/shelf combinations for 
        # which supply is available: [BS, PS], where PS is the number of desired physical items in the warehouse
        idx = torch.argsort(torch.rand(*batch_size, np.prod(self.size_tuple)))[:,:self.num_storage_locations]
        # in order to select only those supply nodes which were sampled in idx, flatten the supply tensor [BS, P*S]. 
        # Select only entries from supply which were sampled in idx and use entries of zeros tensor otherwise. 
        # In the end reshape to [BS, P, S]
        supply = torch.scatter(
            torch.zeros(*batch_size, np.prod(self.size_tuple)), 
            dim=1, 
            index=idx, 
            src=supply.view(*batch_size, -1)
        )
        supply = supply.view(*batch_size, self.num_shelves, self.num_skus)

        assert torch.all((supply>0).sum((1,2)).eq(self.num_storage_locations))

        # make instance feasible by reducing demand to sum of supply for this sku
        demand = torch.minimum(demand, supply.sum(1))
        assert not demand.eq(0).all()

        if self.num_agents is None:
            self.num_agents = torch.ceil(demand.sum(-1) / self.capacity)

        current_location = torch.randint(0, self.num_depots, size=(*batch_size, self.num_agents))
        capacity = torch.full((*batch_size, self.num_agents), fill_value=self.capacity, dtype=torch.float32)
        return TensorDict(
            {
                "demand": demand,
                "supply": supply,
                "coordinates": coordinates,
                "current_location": current_location,
                "remaining_capacity": capacity
            }, 
            batch_size=batch_size
        )
        


    def __call__(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._simulate_batch(batch_size)

