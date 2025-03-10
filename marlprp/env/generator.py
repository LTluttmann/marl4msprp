import os
import torch
import numpy as np
from tensordict import TensorDict

from marlprp.utils.config import EnvParams, LargeEnvParams, EnvParamList


class MSPRPGenerator:

    def __init__(
        self,
        instance_params: EnvParams
    ):

        super(MSPRPGenerator, self).__init__()
        
        self.num_depots = instance_params.num_depots
        self.num_skus = instance_params.num_skus
        self.num_shelves = instance_params.num_shelves
        self.size = self.num_shelves * self.num_skus
        self.num_storage_locations = instance_params.num_storage_locations
        self.capacity = instance_params.capacity

        # get demand params
        self.min_demand = instance_params.min_demand
        self.max_demand = instance_params.max_demand

        self.min_supply = instance_params.min_supply
        if instance_params.max_supply is None:
            avg_demand_per_sku = (self.max_demand + self.min_demand) / 2
            avg_total_supply_per_sku = instance_params.avg_supply_to_demand_ratio * avg_demand_per_sku
            avg_storgage_locations_per_sku = self.num_storage_locations / self.num_skus
            avg_supply_per_storage_loc = max(avg_total_supply_per_sku / avg_storgage_locations_per_sku, 1)
            self.max_supply = np.ceil(avg_supply_per_storage_loc * 2 - self.min_supply).astype("int")
        else:
            self.max_supply = instance_params.max_supply

        self._num_agents = instance_params.num_agents
        self.id = f"{self.num_shelves}s-{self.num_skus}i-{self.num_storage_locations}p"

    def _simulate_batch(self, bs: tuple):
        # simulate supply [BS, P, S]

        supply = torch.randint(
            low=self.min_supply, 
            high=self.max_supply+1, 
            size=(*bs, self.num_shelves, self.num_skus),
            dtype=torch.float32
        )

        # simulate demand [BS, P]; add 1 since max of randint is 'high'-1
        demand = torch.randint(
            low=self.min_demand, 
            high=self.max_demand+1, 
            size=(*bs, self.num_skus),
            dtype=torch.float32
        )

        num_nodes = self.num_shelves + self.num_depots
        # simulate shelf locations as x,y coordinates in a unit circle [BS, S, 2]
        coordinates = torch.rand((*bs, num_nodes, 2))

        # simulate for each batch a series of indices which correspond to the item/shelf combinations for 
        # which supply is available: [BS, PS], where PS is the number of desired physical items in the warehouse
        idx = torch.argsort(torch.rand(*bs, self.size))[:,:self.num_storage_locations]
        # in order to select only those supply nodes which were sampled in idx, flatten the supply tensor [BS, P*S]. 
        # Select only entries from supply which were sampled in idx and use entries of zeros tensor otherwise. 
        # In the end reshape to [BS, P, S]
        supply = torch.scatter(
            torch.zeros(*bs, self.size), 
            dim=1, 
            index=idx, 
            src=supply.view(*bs, -1)
        )
        supply = supply.view(*bs, self.num_shelves, self.num_skus)

        assert torch.all((supply>0).sum((1,2)).eq(self.num_storage_locations))

        # make instance feasible by reducing demand to sum of supply for this sku
        demand = torch.minimum(demand, supply.sum(1))
        assert not demand.eq(0).all()

        if self._num_agents is None:
            num_agents = torch.ceil(demand.sum(-1, keepdim=True) / self.capacity)
            max_num_agents = int(num_agents.max().item())
            agent_pad_mask = num_agents < torch.arange(1, max_num_agents+1).view(1, -1).expand(*bs, max_num_agents)
            num_agents = max_num_agents
        else:
            num_agents = self._num_agents
            agent_pad_mask = torch.full((*bs, num_agents), fill_value=False)
        current_location = torch.randint(0, self.num_depots, size=(*bs, num_agents))
        capacity = torch.full((*bs, num_agents), fill_value=self.capacity, dtype=torch.float32)
        capacity.masked_fill_(agent_pad_mask, 0)
        return TensorDict(
            {
                "demand": demand,
                "supply": supply,
                "coordinates": coordinates,
                "current_location": current_location,
                "init_capacity": capacity,
                "agent_pad_mask": agent_pad_mask
            }, 
            batch_size=bs
        )
        


    def __call__(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        while True:
            td = self._simulate_batch(batch_size)
            if not td["demand"].eq(0).all():
                return td


if __name__ == "__main__":
    params = EnvParams(
        num_agents=None, 
        num_depots=1, 
        num_shelves=50, 
        num_skus=100, 
        num_storage_locations=200, 
        capacity=15
    )
    gen = MSPRPGenerator(params)
    td = gen([20])
    td["init_capacity"] = td["init_capacity"][:,0]
    td["current_location"] = td["current_location"][:,0]
    
    save_dir = os.path.join("data_test/large", gen.id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(td, os.path.join(save_dir, "td_data.pth"))