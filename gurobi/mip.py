import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from itertools import permutations
import itertools
import gurobipy as gp
from gurobipy import GRB
import logging
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class InstanceSolution:
    reward: float
    tour_and_units: Optional[List[Tuple[int, float]]]
    runtime: float
    is_optimal: bool


@dataclass
class SolverInput:
    skus: List[int]
    agents: List[int]
    shelves: List[int]
    nodes: List[int]
    arcs: List[tuple]
    shelves_of_item: Dict[int, list]
    demand_of_item: Dict[int, int]
    capacity: int
    supply_at_shelf: List[int]  
    distances: List[List[float]]
    item_at_node: Dict[int, int]
    storage_loc_shelf_id_map: Dict[int, int]

    def add_vehicle(self):
        self.agents.append(len(self.agents))


@dataclass
class SolverOutput:
    visit: gp.tupledict
    taken_units: gp.tupledict
    max_tour_len: float



@dataclass
class GurobiSolution:
    grb_output: SolverOutput
    grb_input: SolverInput
    grb_model: gp.Model


def solver_model(d: SolverInput) -> Tuple[gp.Model, Tuple[Any, Any]]:

    BigM = d.capacity

    m = gp.Model('MIXED')

    x = {}
    for i in d.nodes:
        for j in d.nodes:
            for k in d.agents:
                x[i,j,k] = m.addVar(vtype=GRB.BINARY, name=f'x[{i},{j},{k}]')

    y = {}
    for s in d.shelves:
        for k in d.agents:
            y[s,k] = m.addVar(lb=0, vtype=GRB.SEMIINT, name=f'y[{s},{k}]')

    # Auxiliary variable to represent the maximum distance
    z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z')

    # Constraints to ensure z is the maximum distance for any vehicle
    for k in d.agents:
        m.addConstr(
            z >= gp.quicksum(x[i, j, k] * d.distances[i, j] for i, j in d.arcs),
            name=f'max_distance_constraint[{k}]'
        )

    # Set objective to minimize the maximum distance
    m.setObjective(z, GRB.MINIMIZE)

    # CONSTRAINTS
    # network flow constraint
    m.addConstrs(gp.quicksum(x[i,j,k] for j in d.nodes if j != i) == gp.quicksum(x[j,i,k] for j in d.nodes if j != i) for i in d.nodes for k in d.agents)
    # max one visit of node and vehicle
    m.addConstrs(gp.quicksum(x[i,j,k] for j in d.nodes) <= 1 for i in d.nodes for k in d.agents)
    # dont take more units than are available at shelf
    m.addConstrs(gp.quicksum(y[i,k] for k in d.agents) <= d.supply_at_shelf[i] for i in d.shelves)
    m.addConstrs(y[j,k] <= BigM * gp.quicksum(x[i,j,k] for i in d.nodes) for j in d.shelves for k in d.agents)
    # m.addConstrs(gp.quicksum(y[p,k] for k in d.agents) <= gp.quicksum(d.supply_at_shelf[j]*x[i,j,k] for i in d.nodes for j in d.shelves_of_item[p] for k in d.agents) for p in d.skus)
    # meet the demand exactely
    m.addConstrs(gp.quicksum(y[i,k] for i in d.shelves_of_item[p] for k in d.agents) == d.demand_of_item[p] for p in d.skus)
    # dont exceed the capacity of the agents
    m.addConstrs(gp.quicksum(y[i,k] for i in d.shelves) <= d.capacity for k in d.agents)
    # dont allow for self-visits
    m.addConstrs(x[i,i,k]==0 for i in d.nodes for k in d.agents)
    # let each vehicle leave the depot once
    m.addConstrs(gp.quicksum(x[0,j,k] for j in d.shelves ) == 1 for k in d.agents)
    # avoid that a node is visited and no units are taken (might happen if distance to node is zero)
    m.addConstrs(x[i,j,k] <= y[j,k] for i in d.nodes for j in d.shelves for k in d.agents)

    m._x = x

    m.Params.LazyConstraints = 1

    return m, (x,y,z)                                                                                                                                                                                                                              


def flatten_data_to_storage_locations(data):
    """the MIP model from Luttmann, Xie (2024) expects a flattened representation, where each sku/shelf combination
    poses a storage location, instead of considering shelves, that may store multiple skus. To this end, the data is
    flattened in this function"""

    num_depots = data["coordinates"].size(0) - data["supply"].size(0)
    depot_loc = data["coordinates"][:num_depots]
    loc = data["coordinates"][num_depots:]
    supply = data["supply"]

    n_shelves, n_items = supply.size()
    valid_nodes = supply.permute(1,0).flatten() > 0
    helper = torch.eye(n_items, n_items).repeat_interleave(n_shelves, dim=0)
    supply_flat_and_tiled = helper * supply.repeat(n_items, 1)
    supply = supply_flat_and_tiled[valid_nodes].view(-1, n_items)
    supply_w_depot = torch.cat((supply.new_zeros((num_depots, n_items)), supply), dim=0)

    loc = loc.unsqueeze(0).expand(n_items,-1,-1)
    loc = loc.flatten(0,1)[valid_nodes].view(-1, 2)
    loc = torch.cat((depot_loc, loc), dim=0)

    shelf_ids = torch.arange(num_depots, n_shelves+num_depots)
    shelf_ids = shelf_ids.view(-1, 1).repeat(1, n_items).permute(1,0).flatten()
    shelf_ids_of_storage_locs = shelf_ids[valid_nodes]
    storage_loc_shelf_id_map = {
        storage_loc+num_depots: shelf_id.item() for storage_loc, shelf_id in enumerate(shelf_ids_of_storage_locs)
    }
    # add depot
    storage_loc_shelf_id_map.update({i: i for i in range(num_depots)})
    return supply_w_depot, loc, storage_loc_shelf_id_map



def transform_data_for_solver_input(data):
    num_nodes = data["coordinates"].size(0)
    num_shelves = data["supply"].size(0)
    num_depots = num_nodes - num_shelves
    data = data.clone().to("cpu")
    
    loads = data["init_capacity"][0]

    sku_ids = data["demand"].nonzero().squeeze(1).numpy()

    supply_w_depot, loc, storage_loc_shelf_id_map = flatten_data_to_storage_locations(data)
    storage_loc_ids, sku_of_storage_loc = map(
        lambda x: x.squeeze(1), supply_w_depot.nonzero().split(1, dim=1)
    )

    # exclude nodes of items with no demand
    valid_locs = [x.item() in sku_ids for x in sku_of_storage_loc]

    # exclude storage positions of skus without demand
    storage_locs = storage_loc_ids[valid_locs].tolist()
    storage_locs_and_depots = list(range(num_depots)) + storage_locs

    distances = torch.cdist(loc, loc)
    # all edges
    edges = [(i,j) for i in storage_locs_and_depots for j in storage_locs_and_depots]  
    # shelves containing item
    N_p = {p: supply_w_depot[:,p].nonzero().squeeze(1).numpy() for p in sku_ids}
    # supply of item / shelf combination
    q = supply_w_depot.sum(1).tolist()
    # demand of item
    r_p = {p: data["demand"][p].item() for p in sku_ids}

    num_tours_lower_bound = np.ceil(sum(list(r_p.values())) / loads).int().item()
    agents = list(range(num_tours_lower_bound))

    item_at_node = {i[0].item(): i[1].item() for i in supply_w_depot.nonzero()}
    
    transformed_data = SolverInput(
        skus=sku_ids,
        agents=agents,
        shelves=storage_locs,
        nodes=storage_locs_and_depots,
        arcs=edges,
        shelves_of_item=N_p,
        demand_of_item=r_p,
        capacity=loads,
        supply_at_shelf=q,
        distances=distances,
        item_at_node=item_at_node,
        storage_loc_shelf_id_map=storage_loc_shelf_id_map
    )

    return transformed_data


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._x)
        edges = gp.tuplelist((i, j, k) for i, j, k in vals.keys() if vals[i, j, k] > 0.5)
        agents = list(set([k for i,j,k in model._x.keys()]))
        for k in agents:
            # edges_of_k = edges.select('*','*',k).select('*','*')
            edges_of_k = gp.tuplelist((i, j) for i, j, k_ in edges if k_ == k)
            tour, is_subtour = subtour(edges_of_k)
            # subtour elimination constraint
            if is_subtour:
                for _k_ in agents:
                    model.cbLazy(
                        gp.quicksum(
                            model._x[i, j, _k_] 
                            for i, j in permutations(tour, 2)
                        ) <= len(tour)-1
                    )           


# Given a tuplelist of edges, find the shortest subtour not containing depot (0)
def subtour(edges):
    if not isinstance(edges, gp.tuplelist): edges = gp.tuplelist(edges)
    unvisited = list(set([i for key in edges for i in key]))
    cycle = range(len(unvisited)+1)  # initial length has 1 more city
    is_subtour = False
    while unvisited:
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            #if current != 0:
            try:
                unvisited.remove(current)
            except:
                pass
            neighbors = [j for i, j in edges.select(current, '*')
                         if j == 0 or j in unvisited]
        if 0 not in thiscycle and len(cycle) > len(thiscycle):
            is_subtour = True
            cycle = thiscycle
    return thiscycle, is_subtour


def get_complete_tour_idx(solution: SolverOutput, d: SolverInput):
    tours_and_units = []
    
    x = solution.visit
    y = solution.taken_units

    for k in d.agents:
        tour_edges = gp.tuplelist((i,j) for i,j,k_ in x if x[i,j,k].X > 0.5 and k_==k)
        tour, is_subtour = subtour(tour_edges)

        assert not is_subtour

        units = []
        for node in tour:
            if node == 0:
                units.append(0.0)
            else:
                item = d.item_at_node[node]
                unit = y.select(node,k)[0].X
                assert unit > 0, f"Zero units are taken at node {node}"
                # sometime gurobi spits out weird floats. Round here
                units.append(round(unit))
                
        tour_and_units = list(zip(tour, units))
        tours_and_units.append(tour_and_units)

    return tours_and_units


def solve(
        instance, 
        timeout=3600, 
        verbose=True, 
        num_threads=None,
        mipfocus=False, 
        add_to_timeout=0, 
        retry_w_mipfocus=False
    ) -> InstanceSolution:
    
    # if timeout == 0:
    #     return InstanceSolution(reward=None, tour_and_units=None, runtime=0), False

    d: SolverInput = transform_data_for_solver_input(instance)

    found_opt = False

    while True:
        
        m, vars = solver_model(d)

        if num_threads is not None:
            # Get the number of threads parameter
            m.setParam("Threads", num_threads)

        log.info("Gurobi is using %s threads" % m.params.Threads)

        if not verbose:
            m.Params.LogToConsole = 0

        if timeout is not None:
            m.setParam('TimeLimit', timeout)

        if mipfocus:
            m.setParam('MIPFocus', 1)

        x,y,z = vars
        m.optimize(subtourelim)
        
        # check if a solution was found, otherwise raise time limit, retry with mipfocus or return empty solution
        if not m.SolCount > 0:

            if retry_w_mipfocus:
                return solve(instance, timeout, mipfocus=True, add_to_timeout=0, retry_w_mipfocus=False)
            
            elif add_to_timeout > 0:
                log.info(f"No solution was found. Raise timelimit by {add_to_timeout} seconds")
                timeout += add_to_timeout
                continue

            else:
                # add empty solution
                return InstanceSolution(reward=None, tour_and_units=None, runtime=m.Runtime), found_opt
            
        elif m.MIPGap > 1e-5: # use small threshold to avoid weird float behavior
            log.info("Preempting...Found non-optimal solution with obj. value %s and gap of %s percent" % 
                    (round(m.ObjVal,2), round(m.MIPGap*100,2)))
            break

        else:
            found_opt = True
            log.info("Found optimal solution with obj. value: %s" % m.ObjVal)
            break
    
    m.update()
    x=gp.tupledict(x)
    y=gp.tupledict(y)
    obj = z.X
    sol = SolverOutput(x,y,obj)
    # infer total tour information from sovler solution
    tours_and_units = get_complete_tour_idx(sol, d)
    tours_and_units_translated = [
        [(d.storage_loc_shelf_id_map[node], units) for node, units in tour] 
        for tour in tours_and_units
    ]
    tests(x, y, tours_and_units, d, instance, obj)
    # return standardized solution output
    sol = InstanceSolution(m.ObjVal, tours_and_units_translated, m.Runtime, found_opt) 
    return sol


def tests(x, y, tour_and_units, d:SolverInput, instance, obj):
    tours = [[x[0] for x in tu] for tu in tour_and_units]
    assert np.isclose(max([get_distance_of_tour(tour, d.distances) for tour in tours]) , obj)

    # test whether no subtours exists
    for k in d.agents:
        selected = gp.tuplelist((i,j) for i,j,k_ in x if x[i,j,k].X > 0.5 and k_==k)
        _, is_subtour = subtour(selected)

        assert not is_subtour

    # test whether capacity limit is not exceeded
    for k in d.agents:

        load = sum([round(i.X) for i in y.select('*',k)])
        log.info(f"utilization of vehicle {k} is {load/d.capacity*100}%")
        assert load <= d.capacity, k

    # test whether demand is exactly met
    for p in d.skus:
        units_at_shelves = [y.select(i,'*') for i in d.shelves_of_item[p]]
        assert sum([round(i.X) for l in units_at_shelves for i in l]) == d.demand_of_item[p], p

    # test whether supply of shelves is not exceeded
    taken_units = pd.DataFrame(list(itertools.chain.from_iterable(tour_and_units))).groupby(0).agg("sum")
    supply = pd.DataFrame(zip(list(range(len(d.supply_at_shelf))), d.supply_at_shelf))
    jj = taken_units.reset_index().merge(supply, on=0, how='left')
    assert all(jj.iloc[:,1] <= jj.iloc[:,2])

    # test that we did not mess up the indices of skus, shelves, storage positions...
    for tour in tours:
        picked_skus = [d.item_at_node[node] for node in tour if node in d.item_at_node]
        assert all([sku in d.skus for sku in picked_skus])

def get_distance_of_tour(tour, distances):
    ds = []
    for (i,j) in zip(tour, tour[1:]):
        ds.append(distances[i,j])
    return sum(ds)