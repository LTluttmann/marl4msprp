import torch
import torch.nn.functional as F
from tensordict import tensorclass


@tensorclass
class MSPRPState:
    supply: torch.Tensor
    demand: torch.Tensor
    coordinates: torch.Tensor
    current_location: torch.Tensor
    remaining_capacity: torch.Tensor
    agent_pad_mask: torch.Tensor
    tour_length: torch.Tensor = None
    packing_items: torch.Tensor = None
    init_capacity: torch.Tensor = None
    zero_units_taken: torch.Tensor = None

    @classmethod
    def initialize(
        cls,
        supply: torch.Tensor,
        demand: torch.Tensor,
        coordinates: torch.Tensor,
        current_location: torch.Tensor,
        remaining_capacity: torch.Tensor,
        agent_pad_mask: torch.Tensor = None,
        tour_length: torch.Tensor = None,
        packing_items: torch.Tensor = None,
        zero_units_taken: torch.Tensor = None,
    ) -> "MSPRPState":
        batch_size = supply.size(0)
        return cls(
            supply=supply,
            demand=demand,
            coordinates=coordinates,
            current_location=current_location,
            remaining_capacity=remaining_capacity,
            agent_pad_mask=agent_pad_mask,
            tour_length=tour_length,
            packing_items=packing_items,
            zero_units_taken=zero_units_taken,
            batch_size=[batch_size],
            device=supply.device
        )
    
    def __post_init__(self):
        if self.tour_length is None:
            self.tour_length = torch.zeros(
                (*self.batch_size, self.num_agents), 
                dtype=torch.float32, 
                device=self.device
            )
            
        if self.packing_items is None:
            self.packing_items = torch.zeros(
                (*self.batch_size, self.num_depots), 
                dtype=torch.float32, 
                device=self.device
            )

        if self.init_capacity is None:
            self.init_capacity = self.remaining_capacity.clone()

        if self.agent_pad_mask is None:
            self.agent_pad_mask = torch.full(
                (*self.batch_size, self.num_agents),
                fill_value=False,
                dtype=torch.bool,
                device=self.device
            )
        if self.zero_units_taken is None:
            self.zero_units_taken = torch.zeros(self.batch_size, device=self.device)

    @property
    def capacity(self):
        return self.init_capacity.max().item()

    @property
    def num_shelves(self):
        return self.supply.size(1)

    @property
    def num_nodes(self):
        return self.coordinates.size(1)

    @property
    def num_skus(self):
        return self.demand.size(1)

    @property
    def num_depots(self):
        return self.num_nodes - self.num_shelves

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
    
    @property
    def current_loc_ohe(self):
        return F.one_hot(
            self.current_location, 
            num_classes=self.num_shelves + self.num_depots
        )