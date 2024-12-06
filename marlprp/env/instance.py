import torch
from tensordict import tensorclass
from dataclasses import field


@tensorclass
class MSPRPState:
    _num_depots: str
    supply: torch.Tensor
    demand: torch.Tensor
    coordinates: torch.Tensor
    current_location: torch.Tensor
    remaining_capacity: torch.Tensor
    tour_length: torch.Tensor = None
    packing_items: torch.Tensor = None

    @classmethod
    def initialize(
        cls,
        num_depots: int,
        supply: torch.Tensor,
        demand: torch.Tensor,
        coordinates: torch.Tensor,
        current_location: torch.Tensor,
        remaining_capacity: torch.Tensor,
        tour_length = None,
        packing_items = None
    ):
        batch_size = supply.size(0)
        return cls(
            _num_depots=str(num_depots),
            supply=supply,
            demand=demand,
            coordinates=coordinates,
            current_location=current_location,
            remaining_capacity=remaining_capacity,
            batch_size=[batch_size],
            tour_length=tour_length,
            packing_items=packing_items,
        )
    
    def __post_init__(self):
        if self.tour_length is None:
            self.tour_length = torch.zeros((*self.batch_size, self.num_agents), dtype=torch.float32, device=self.device)
        if self.packing_items is None:
            self.packing_items = torch.zeros((*self.batch_size, self.num_depots), dtype=torch.float32, device=self.device)
    
    @property
    def num_shelves(self):
        return self.coordinates.size(1) - self.num_depots

    @property
    def num_skus(self):
        return self.demand.size(1)

    @property
    def num_depots(self):
        return int(self._num_depots)

    @property
    def num_agents(self):
        return self.current_location.size(-1)

    @property
    def supply_w_depot(self):
        bs = self.batch_size
        depots = self.supply.new_zeros(size=(*bs, self.num_depots, self.num_skus))
        return torch.cat((depots, self.supply), dim=1)

    @property
    def shelf_locations(self):
        return self.coordinates[:, self.num_depots:]
    
    @property
    def depot_locations(self):
        return self.coordinates[:, :self.num_depots]
    
    def agent_at_depot(self, agent: torch.Tensor = None):
        # if we have 5 nodes, 2 depots and 3 shelves, the order is <d d s s s>
        if agent is None:
            return self.current_location < self.num_depots
        else: 
            return self.current_location.gather(1, agent[:, None]).squeeze(1) < self.num_depots
    
    @property
    def done(self):
        # (bs)
        return (self.demand.le(1e-5).all(-1) & self.agent_at_depot().all(-1))