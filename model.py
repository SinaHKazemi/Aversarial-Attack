from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB





@dataclass(frozen=True)
class HouseParameters:
    # check the length of the demand and pv_availability
    # f"The length of the demand vector ({demand_len}) and PV availability vector ({pv_availability_len}) are not the same."
    life_time: int
    price_PV: float
    price_battery: float
    cost_buy: float
    sell_price: float
    total_demand: float
    demands: list[float]
    PV_availabilities: list[float]
    
    def cost_PV(self):
        return self.price_PV/self.life_time
    
    def cost_battery(self):
        return self.price_battery/self.life_time
    
    @property
    def hours_num(self):
        return len(self.demands)

@dataclass(frozen=True)
class AttackParams:    
    lb: float|list[float] = - GRB.INFINITY
    ub: float|list[float] = GRB.INFINITY
    norm: int = 1 # norm of the objective function, 1 results in a linear model and 2 results in a convex quadratic one
    total_delta_ub: float = GRB.INFINITY

class HouseModel():
    def __init__(self, house_params: HouseParameters, attack_params: AttackParams):       
        self.house_params = house_params
        self.attack_params = attack_params
        self.model = gp.Model("HouseModel")
        self.vars = {}
        self.constrs = {}

    def _add_vars(self) -> None:
        # upper level
        self.model.addVars(self.house_params.hours_num, vtype= GRB.CONTINUOUS, name="delta")
        self.model.addVar(name="upper_level_obj", vtype= GRB.CONTINUOUS)
        
        # primal
        self.vars["primal"] = {}
        self.vars["primal"]["energy_PV"] = self.model.addVars(self.house_params.hours_num, name="energy_PV")
        self.vars["primal"]["energy_battery"] = self.model.addVars(self.house_params.hours_num, name="energy_battery")
        self.vars["primal"]["energy_battery_in"] = self.model.addVars(self.house_params.hours_num, name="energy_battery_in")
        self.vars["primal"]["energy_battery_out"] = self.model.addVars(self.house_params.hours_num, name="energy_battery_out")
        self.vars["primal"]["energy_buy"] = self.model.addVars(self.house_params.hours_num, name="energy_buy")
        self.vars["primal"]["energy_sell"] = self.model.addVars(self.house_params.hours_num, name="energy_sell")
        self.vars["primal"]["capacity_battery"] = self.model.addVar(name="capacity_battery")
        self.vars["primal"]["capacity_PV"] = self.model.addVar(name="capacity_PV")

        self.vars["primal"]["obj"] = self.model.addVar(name="primal_obj", vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        
        # dual
        self.vars["dual"] = {} 
        self.vars["dual"]["limit_battery"] = self.model.addVars(self.house_params.hours_num, name="limit_battery", lb=-GRB.INFINITY, ub=0)
        self.vars["dual"]["limit_PV"] = self.model.addVars(self.house_params.hours_num, name="limit_PV", lb=-GRB.INFINITY, ub=0)
        self.vars["dual"]["eq_battery"] = self.model.addVars(self.house_params.hours_num, name="eq_battery", lb=-GRB.INFINITY)
        self.vars["dual"]["eq_demand"] = self.model.addVars(self.house_params.hours_num, name="eq_demand", lb=-GRB.INFINITY)

        self.vars["dual"]["obj"] = self.model.addVar(name="dual_obj", lb=-GRB.INFINITY)
        
        # auxiliary variables for complementary slackness
        # positive variables and positive slacks of the constraints

    
    def _add_upper_level_constrs(self) -> None:
        pass
    
    def _add_primal_constrs(self) -> None:
        self.constrs["primal"] = {}
        self.constrs["primal"]["eq_energy"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_buy"][i] -
                self.vars["primal"]["energy_sell"][i] +
                self.vars["primal"]["energy_battery_out"][i] -
                self.vars["primal"]["energy_battery_in"][i] +
                self.vars["primal"]["energy_PV"][i] == 
                self.house_params.demands[i] * self.house_params.total_demand  for i in range(self.house_params.hours_num)
            ),
            name="eq_energy"
        )

        self.constrs["primal"]["eq_battery"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_battery"][range(self.house_params.hours_num)[i-1]] +
                self.vars["primal"]["energy_battery_in"][i] -
                self.vars["primal"]["energy_battery_out"][i] == 
                self.vars["primal"]["energy_battery"][i]  for i in range(self.house_params.hours_num)
            ),
            name="eq_battery"
        )

        self.constrs["primal"]["capacity_PV"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_PV"][i] <= 
                self.vars["primal"]["capacity_PV"] * self.house_params.PV_availabilities[i] for i in range(self.house_params.hours_num)
            ),
            name="capacity_PV"
        )

        self.constrs["primal"]["capacity_battery"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_battery"][i] <= 
                self.vars["primal"]["capacity_battery"] for i in range(self.house_params.hours_num)
            ),
            name="capacity_battery"
        )

        # objective function
        self.addConstr(
            self.house_params.cost_PV * self.vars["capacity_PV"] +
            self.house_params.cost_battery * self.vars["capacity_battery"] +
            self.house_params.cost_buy * sum(self.vars["primal"]["energy_buy"][i] for i in range(self.house_params.hours_num)) -
            self.house_params.sell_price * sum(self.vars["primal"]["energy_sell"][i] for i in range(self.house_params.hours_num)) ==
            self.vars["primal"]["obj"]
        )

    def _add_dual_constrs(self) -> None:
        self.constrs["dual"] = {}

        self.constrs["dual"]["energy_buy"] = self.model.addConstrs(
            (
                self.vars["dual"]["eq_demand"][i] <= 
                self.house_params.cost_buy for i in range(self.house_params.hours_num)
            ),
            name="energy_buy"
        )
        self.constrs["dual"]["energy_sell"] = self.model.addConstrs(
            (
                - self.vars["dual"]["eq_demand"][i] <= 
                - self.house_params.cost_sell for i in range(self.house_params.hours_num)
            ),
            name="energy_sell"
        )
        self.constrs["dual"]["energy_battery_out"] = self.model.addConstrs(
            (self.vars["dual"]["eq_demand"][i] - self.vars["dual"]["eq_battery"][i]<= 0 for i in range(self.house_params.hours_num)),
            name="energy_battery_out"
        )
        self.constrs["dual"]["energy_battery_in"] = self.model.addConstrs(
            - (self.vars["dual"]["eq_demand"][i] + self.vars["dual"]["eq_battery"][i]<= 0 for i in range(self.house_params.hours_num)),
            name="energy_battery_in"
        )
        self.constrs["dual"]["energy_battery"] = self.model.addConstrs(
            (
                self.vars["dual"]["eq_battery"][i] -
                self.vars["dual"]["eq_battery"][range(self.house_params.hours_num)[i-1]] + 
                self.vars["dual"]["limit_battery"][range(self.house_params.hours_num)[i-1]] <= 
                0 for i in range(self.house_params.hours_num)
            ),
            name="energy_battery"
        )
        self.constrs["dual"]["energy_PV"] = self.model.addConstrs(
            (
                self.vars["dual"]["eq_demand"][i] +
                self.vars["dual"]["limit_PV"][i] <= 
                0 for i in range(self.house_params.hours_num)
            ),
            name="energy_PV"
        )
        self.constrs["dual"]["capacity_battery"] = self.model.addConstr(
            sum(-self.vars["dual"]["limit_battery"][i] for i in range(self.house_params.hours_num))  <=  self.house_params.cost_battery,
            name="capacity_battery"
        )
        self.constrs["dual"]["capacity_PV"] = self.model.addConstr(
            sum(-self.vars["dual"]["limit_PV"][i] * self.house_params.PV_availabilities[i] for i in range(self.house_params.hours_num))  <=  self.house_params.cost_PV,
            name="capacity_PV"
        )
        
        # objective function

    
    def solve(self) -> None:
        pass
    
    def get_upper_level_values(self) -> dict[str, list[float]]:
        pass
    
    def get_primal_values(self) -> dict[str, list[float]]:
        pass
    
    def get_dual_values(self) -> dict[str, list[float]]:
        pass

