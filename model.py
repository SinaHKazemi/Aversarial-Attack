from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
import numpy as np

@dataclass(frozen=True)
class HouseParams:
    life_time: int
    price_PV: float
    price_battery: float
    cost_buy: float
    sell_price: float
    total_demand: float
    demands: list[float]
    PV_availabilities: list[float]
    
    @property
    def cost_PV(self):
        return self.price_PV/self.life_time
    
    @property
    def cost_battery(self):
        return self.price_battery/self.life_time
    
    @property
    def hours_num(self):
        return len(self.demands)

@dataclass(frozen=True)
class AttackParams:
    capacity_battery: float|None = None
    capacity_PV: float|None = None
    lb: float|list[float] = - GRB.INFINITY
    ub: float|list[float] = GRB.INFINITY
    norm: int = 1 # norm of the objective function, 1 results in a linear model and 2 results in a convex quadratic one
    total_delta_ub: float = GRB.INFINITY
    

@dataclass(frozen=True)
class PADM_Params:
    initial_mu: float = 1
    increase_factor: float = 2
    max_penalty_iter: int = 15
    max_stationary_iter: int = 200
    stationary_error: float = 1e-4
    penalty_error: float = 1e-4

class HouseModel():
    def __init__(self, house_params: HouseParams, attack_params: AttackParams):       
        self.house_params = house_params
        self.attack_params = attack_params
        self.model = gp.Model("HouseModel")
        self.vars = {}
        self.obj = {} # to keep objective function expressions
        self.constrs = {}
        self.constrs["fix"] = {} # to keep the variable fixing constraints

    def add_vars(self) -> None:
        # upper level
        self.vars["upper_level"] = {}
        self.vars["upper_level"]["delta"] = self.model.addVars(self.house_params.hours_num, vtype= GRB.CONTINUOUS, name="delta",lb=self.attack_params.lb,ub=self.attack_params.ub)
        self.vars["upper_level"]["abs"] = self.model.addVars(self.house_params.hours_num, vtype= GRB.CONTINUOUS, name="abs",lb=-GRB.INFINITY)
        
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
        
        # dual
        self.vars["dual"] = {} 
        self.vars["dual"]["limit_battery"] = self.model.addVars(self.house_params.hours_num, name="limit_battery", lb=-GRB.INFINITY, ub=0)
        self.vars["dual"]["limit_PV"] = self.model.addVars(self.house_params.hours_num, name="limit_PV", lb=-GRB.INFINITY, ub=0)
        self.vars["dual"]["eq_battery"] = self.model.addVars(self.house_params.hours_num, name="eq_battery", lb=-GRB.INFINITY)
        self.vars["dual"]["eq_demand"] = self.model.addVars(self.house_params.hours_num, name="eq_demand", lb=-GRB.INFINITY)
        
        # auxiliary variables for complementary slackness
        # positive variables and positive slacks of the constraints
        self.vars["aux"] = {}
        self.vars["aux"]["limit_PV"] = self.model.addVars(self.house_params.hours_num, name="aux_limit_PV")
        self.vars["aux"]["limit_battery"] = self.model.addVars(self.house_params.hours_num, name="aux_limit_battery")
        self.vars["aux"]["slack_limit_PV"] = self.model.addVars(self.house_params.hours_num, name="slack_limit_PV")
        self.vars["aux"]["slack_limit_battery"] = self.model.addVars(self.house_params.hours_num, name="slack_limit_battery")
        self.vars["aux"]["slack_energy_buy"] = self.model.addVars(self.house_params.hours_num, name="slack_energy_buy")
        self.vars["aux"]["slack_energy_sell"] = self.model.addVars(self.house_params.hours_num, name="slack_energy_sell")
        self.vars["aux"]["slack_energy_battery_out"] = self.model.addVars(self.house_params.hours_num, name="slack_energy_battery_out")
        self.vars["aux"]["slack_energy_battery_in"] = self.model.addVars(self.house_params.hours_num, name="slack_energy_battery_in")
        self.vars["aux"]["slack_energy_battery"] = self.model.addVars(self.house_params.hours_num, name="slack_energy_battery")
        self.vars["aux"]["slack_energy_PV"] = self.model.addVars(self.house_params.hours_num, name="slack_energy_PV")
        self.vars["aux"]["slack_capacity_PV"] = self.model.addVar(name="slack_capacity_PV")
        self.vars["aux"]["slack_capacity_battery"] = self.model.addVar(name="slack_capacity_battery")

        # binary variables for complementary slackness constraints
        self.vars["cs"] = {}
        self.vars["cs"]["limit_PV"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_limit_PV")
        self.vars["cs"]["limit_battery"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_limit_battery")
        self.vars["cs"]["energy_buy"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_energy_buy")
        self.vars["cs"]["energy_sell"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_energy_sell")
        self.vars["cs"]["energy_battery_out"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_energy_battery_out")
        self.vars["cs"]["energy_battery_in"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_energy_battery_in")
        self.vars["cs"]["energy_battery"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_energy_battery")
        self.vars["cs"]["energy_PV"] = self.model.addVars(self.house_params.hours_num, vtype=GRB.BINARY, name="cs_energy_PV")
        self.vars["cs"]["capacity_PV"] = self.model.addVar(vtype=GRB.BINARY, name="cs_limit_PV")
        self.vars["cs"]["capacity_battery"] = self.model.addVar(vtype=GRB.BINARY, name="cs_limit_battery")
        
    def add_upper_level_constrs(self) -> None:
        # constrainsts
        self.constrs["upper_level"] = {}
        self.constrs["upper_level"]["demand_change"] = self.model.addConstr(
            sum(self.vars["upper_level"]["delta"][i] * self.house_params.demands[i] for i in range(self.house_params.hours_num)) == 0,
            name = "demand_change"
        )
        self.constrs["upper_level"]["abs"] = self.model.addConstrs(
            (self.vars["upper_level"]["abs"][i] == gp.abs_(self.vars["upper_level"]["delta"][i]) for i in range(self.house_params.hours_num) ),
            name = "abs"
        )
        
        if self.attack_params.capacity_battery is not None:
            self.constrs["upper_level"]["capacity_battery"] = self.model.addConstr(
                self.attack_params.capacity_battery == self.vars["primal"]["capacity_battery"],
                name = "capacity_battery_lb"
            )
        
        if self.attack_params.capacity_PV is not None:
            self.constrs["upper_level"]["capacity_PV"] = self.model.addConstr(
                self.attack_params.capacity_PV == self.vars["primal"]["capacity_PV"],
                name = "capacity_PV"
            )
        
        # objective
        self.obj["upper_level"] = sum(self.vars["upper_level"]["abs"][i]  for i in range(self.house_params.hours_num))
    
    def add_primal_constrs(self) -> None:
        self.constrs["primal"] = {}
        self.constrs["primal"]["eq_demand"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_buy"][i] -
                self.vars["primal"]["energy_sell"][i] +
                self.vars["primal"]["energy_battery_out"][i] -
                self.vars["primal"]["energy_battery_in"][i] +
                self.vars["primal"]["energy_PV"][i] == 
                self.house_params.demands[i] * self.house_params.total_demand * (1 + self.vars["upper_level"]["delta"][i]) for i in range(self.house_params.hours_num)
            ),
            name="eq_demand"
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

        self.constrs["primal"]["limit_PV"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_PV"][i] <= 
                self.vars["primal"]["capacity_PV"] * self.house_params.PV_availabilities[i] for i in range(self.house_params.hours_num)
            ),
            name="limit_PV"
        )

        self.constrs["primal"]["limit_battery"] = self.model.addConstrs(
            (
                self.vars["primal"]["energy_battery"][i] <= 
                self.vars["primal"]["capacity_battery"] for i in range(self.house_params.hours_num)
            ),
            name="limit_battery"
        )

        # objective function
        self.obj["primal"] = (
            self.house_params.cost_PV * self.vars["primal"]["capacity_PV"] +
            self.house_params.cost_battery * self.vars["primal"]["capacity_battery"] +
            self.house_params.cost_buy * sum(self.vars["primal"]["energy_buy"][i] for i in range(self.house_params.hours_num)) -
            self.house_params.sell_price * sum(self.vars["primal"]["energy_sell"][i] for i in range(self.house_params.hours_num)) 
        )

    def add_dual_constrs(self) -> None:
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
                - self.house_params.sell_price for i in range(self.house_params.hours_num)
            ),
            name="energy_sell"
        )
        self.constrs["dual"]["energy_battery_out"] = self.model.addConstrs(
            (self.vars["dual"]["eq_demand"][i] - self.vars["dual"]["eq_battery"][i]<= 0 for i in range(self.house_params.hours_num)),
            name="energy_battery_out"
        )
        self.constrs["dual"]["energy_battery_in"] = self.model.addConstrs(
            (- self.vars["dual"]["eq_demand"][i] + self.vars["dual"]["eq_battery"][i]<= 0 for i in range(self.house_params.hours_num)),
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
        self.obj["dual"] = sum(self.vars["dual"]["eq_demand"][i] * self.house_params.demands[i] * self.house_params.total_demand * (1 + self.vars["upper_level"]["delta"][i]) for i in range(self.house_params.hours_num))

    def add_aux_constrs(self) -> None:
        self.constrs["aux"] = {}
        self.constrs["aux"]["limit_PV"] = self.model.addConstrs(
            (
                self.vars["aux"]["limit_PV"][i] == 
                - self.vars["dual"]["limit_PV"][i] for i in range(self.house_params.hours_num)
            ),
            name="aux_limit_PV"
        )

        self.constrs["aux"]["limit_battery"] = self.model.addConstrs(
            (
                self.vars["aux"]["limit_battery"][i] == 
                - self.vars["dual"]["limit_battery"][i] for i in range(self.house_params.hours_num)
            ),
            name="aux_limit_battery"
        )

        self.constrs["aux"]["slack_limit_PV"] = self.model.addConstrs(
            (
                self.vars["primal"]["capacity_PV"] * self.house_params.PV_availabilities[i] -
                self.vars["primal"]["energy_PV"][i] == 
                self.vars["aux"]["slack_limit_PV"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_limit_PV"
        )

        self.constrs["aux"]["slack_limit_battery"] = self.model.addConstrs(
            (
                self.vars["primal"]["capacity_battery"] -
                self.vars["primal"]["energy_battery"][i] == 
                self.vars["aux"]["slack_limit_battery"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_limit_battery"
        )

        self.constrs["aux"]["slack_energy_buy"] = self.model.addConstrs(
            (
                self.house_params.cost_buy - self.vars["dual"]["eq_demand"][i] == 
                self.vars["aux"]["slack_energy_buy"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_energy_buy"
        )

        self.constrs["aux"]["slack_energy_sell"] = self.model.addConstrs(
            (
                self.vars["dual"]["eq_demand"][i] - self.house_params.sell_price == 
                self.vars["aux"]["slack_energy_sell"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_energy_sell"
        )

        self.constrs["aux"]["slack_energy_battery_out"] = self.model.addConstrs(
            (
                self.vars["dual"]["eq_battery"][i] - self.vars["dual"]["eq_demand"][i] == 
                self.vars["aux"]["slack_energy_battery_out"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_energy_battery_out"
        )

        self.constrs["aux"]["slack_energy_battery_in"] = self.model.addConstrs(
            (
                self.vars["dual"]["eq_demand"][i] - self.vars["dual"]["eq_battery"][i] == 
                self.vars["aux"]["slack_energy_battery_in"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_energy_battery_in"
        )

        self.constrs["aux"]["slack_energy_battery"] = self.model.addConstrs(
            (
                - self.vars["dual"]["eq_battery"][i] +
                self.vars["dual"]["eq_battery"][range(self.house_params.hours_num)[i-1]] - 
                self.vars["dual"]["limit_battery"][range(self.house_params.hours_num)[i-1]] == 
                self.vars["aux"]["slack_energy_battery"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_energy_battery"
        )

        self.constrs["aux"]["slack_energy_PV"] = self.model.addConstrs(
            (
                - self.vars["dual"]["eq_demand"][i] -
                self.vars["dual"]["limit_PV"][i] == 
                self.vars["aux"]["slack_energy_PV"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_energy_PV"
        )

        self.constrs["aux"]["slack_capacity_battery"] = self.model.addConstr(
            (
                sum(self.vars["dual"]["limit_battery"][i] for i in range(self.house_params.hours_num)) +
                self.house_params.cost_battery == 
                self.vars["aux"]["slack_capacity_battery"]
            ),
            name="slack_capacity_battery"
        )

        self.constrs["aux"]["slack_capacity_PV"] = self.model.addConstr(
            (
                sum(self.vars["dual"]["limit_PV"][i] * self.house_params.PV_availabilities[i] for i in range(self.house_params.hours_num)) +
                self.house_params.cost_PV == 
                self.vars["aux"]["slack_capacity_PV"]
            ),
            name="slack_capacity_PV"
        )

    def add_bigM_constrs(self, M:float) -> None:
        self.constrs["bigM"] = {}

        self.constrs["bigM"]["limit_PV"] = {}
        self.constrs["bigM"]["limit_PV"]["var"] = self.model.addConstrs(
            (self.vars["aux"]["limit_PV"][i] <= self.vars["cs"]["limit_PV"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_PV_var"
        )
        self.constrs["bigM"]["limit_PV"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_limit_PV"][i] <= (1 - self.vars["cs"]["limit_PV"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_PV_con"
        )

        self.constrs["bigM"]["limit_battery"] = {}
        self.constrs["bigM"]["limit_battery"]["var"] = self.model.addConstrs(
            (self.vars["aux"]["limit_battery"][i] <= self.vars["cs"]["limit_battery"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_battery_var"
        )
        self.constrs["bigM"]["limit_battery"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_limit_battery"][i] <= (1 - self.vars["cs"]["limit_battery"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_battery_con"
        )

        self.constrs["bigM"]["energy_buy"] = {}
        self.constrs["bigM"]["energy_buy"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_buy"][i] <= self.vars["cs"]["energy_buy"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_buy_var"
        )
        self.constrs["bigM"]["energy_buy"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_buy"][i] <= (1 - self.vars["cs"]["energy_buy"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_buy_con"
        )

        self.constrs["bigM"]["energy_sell"] = {}
        self.constrs["bigM"]["energy_sell"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_sell"][i] <= self.vars["cs"]["energy_sell"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_sell_var"
        )
        self.constrs["bigM"]["energy_sell"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_sell"][i] <= (1 - self.vars["cs"]["energy_sell"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_sell_con"
        )

        self.constrs["bigM"]["energy_battery_out"] = {}
        self.constrs["bigM"]["energy_battery_out"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_battery_out"][i] <= self.vars["cs"]["energy_battery_out"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_out_var"
        )
        self.constrs["bigM"]["energy_battery_out"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_battery_out"][i] <= (1 - self.vars["cs"]["energy_battery_out"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_out_con"
        )
        
        self.constrs["bigM"]["energy_battery_in"] = {}
        self.constrs["bigM"]["energy_battery_in"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_battery_in"][i] <= self.vars["cs"]["energy_battery_in"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_in_var"
        )
        self.constrs["bigM"]["energy_battery_in"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_battery_in"][i] <= (1 - self.vars["cs"]["energy_battery_in"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_in_con"
        )

        self.constrs["bigM"]["energy_battery"] = {}
        self.constrs["bigM"]["energy_battery"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_battery"][range(self.house_params.hours_num)[i-1]] <= self.vars["cs"]["energy_battery"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_var"
        )
        self.constrs["bigM"]["energy_battery"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_battery"][i] <= (1 - self.vars["cs"]["energy_battery"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_con"
        )

        self.constrs["bigM"]["energy_PV"] = {}
        self.constrs["bigM"]["energy_PV"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_PV"][i] <= self.vars["cs"]["energy_PV"][i] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_PV_var"
        )
        self.constrs["bigM"]["energy_PV"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_PV"][i] <= (1 - self.vars["cs"]["energy_PV"][i]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_PV_con"
        )

        self.constrs["bigM"]["capacity_battery"] = {}
        self.constrs["bigM"]["capacity_battery"]["var"] = self.model.addConstr(
            self.vars["primal"]["capacity_battery"] <= self.vars["cs"]["capacity_battery"] * M ,
            name="bigM_capacity_battery_var"
        )
        self.constrs["bigM"]["capacity_battery"]["con"] = self.model.addConstr(
            self.vars["aux"]["slack_capacity_battery"] <= (1 - self.vars["cs"]["capacity_battery"]) * M ,
            name="bigM_capacity_battery_con"
        )

        self.constrs["bigM"]["capacity_PV"] = {}
        self.constrs["bigM"]["capacity_PV"]["var"] = self.model.addConstr(
            self.vars["primal"]["capacity_PV"] <= self.vars["cs"]["capacity_PV"] * M ,
            name="bigM_capacity_PV_var"
        )
        self.constrs["bigM"]["capacity_PV"]["con"] = self.model.addConstr(
            self.vars["aux"]["slack_capacity_PV"] <= (1 - self.vars["cs"]["capacity_PV"]) * M ,
            name="bigM_capacity_PV_con"
        )
        self.model.setParam("IntFeasTol", 1e-9)
        
    def add_sos_constrs(self) -> None:
        self.constrs["sos"] = {}
        self.constrs["sos"]["limit_PV"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["limit_PV"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["aux"]["limit_PV"][i],self.vars["aux"]["slack_limit_PV"][i]])
            )
        self.constrs["sos"]["limit_battery"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["limit_battery"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["aux"]["limit_battery"][i],self.vars["aux"]["slack_limit_battery"][i]])
            )
        
        self.constrs["sos"]["energy_buy"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["energy_buy"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_buy"][i],self.vars["aux"]["slack_energy_buy"][i]])
            )
        
        self.constrs["sos"]["energy_sell"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["energy_sell"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_sell"][i],self.vars["aux"]["slack_energy_sell"][i]])
            )
        
        self.constrs["sos"]["energy_battery_out"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["energy_sell"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_battery_out"][i],self.vars["aux"]["slack_energy_battery_out"][i]])
            )
        

        self.constrs["sos"]["energy_battery_in"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["energy_battery_in"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_battery_in"][i],self.vars["aux"]["slack_energy_battery_in"][i]])
            )
        
        self.constrs["sos"]["energy_battery"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["energy_battery"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_battery"][range(self.house_params.hours_num)[i-1]],self.vars["aux"]["slack_energy_battery"][i]])
            )
        
        self.constrs["sos"]["energy_PV"] = []
        for i in range(self.house_params.hours_num):
            self.constrs["sos"]["energy_PV"].append(
                self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_PV"][i],self.vars["aux"]["slack_energy_PV"][i]])
            )
        
        self.constrs["sos"]["capacity_battery"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["capacity_battery"],self.vars["aux"]["slack_capacity_battery"]])
        self.constrs["sos"]["capacity_PV"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["capacity_PV"],self.vars["aux"]["slack_capacity_PV"]])

        # https://www.gurobi.com/documentation/9.5/refman/presos1encoding.html#parameter:PreSOS1Encoding
        # to shut off the reformulation PreSOS1BigM parameter should be set to zero
        self.model.setParam("PreSOS1BigM", 0)

    def add_valid_ineq_constr(self, ub:list[float]) -> None:
        self.constrs["valid_ineq"] = self.model.addConstr(
            self.obj["primal"] <= sum(self.vars["dual"]["eq_demand"][i] * self.house_params.demands[i] * self.house_params.total_demand * (1 + ub[i]) for i in range(self.house_params.hours_num)),
            name="valid_ineq"
        )

    def fix_vars(self, name: str, fix_value:float|None = None) -> None:
        var_list = []
        for value in self.vars[name].values():
            if isinstance(value,gp.tupledict):
                var_list.extend(list(value.values()))
            elif isinstance(value,gp.Var):
                var_list.append(value)
            else:
                raise Exception("it should not happen")
        constr_list = []
        self.constrs["fix"][name] = constr_list
        for var in var_list:
            if fix_value is None:
                constr_list.append(self.model.addConstr(var.X == var))
            else:
                constr_list.append(self.model.addConstr(fix_value == var))

    def release_vars(self, name:str) -> None:
        for con in self.constrs["fix"][name]:
            self.model.remove(con)
        del self.constrs["fix"][name]
    
    def set_obj(self, name:str) -> None:
        sense = None
        if name in ("primal", "upper_level", "PADM"):
            sense = GRB.MINIMIZE
        elif name in ("dual",):
            sense = GRB.MAXIMIZE
        else:
            raise Exception(f"{name} objective function isn't defined")
        self.model.setObjective(self.obj[name], sense)
    
    def set_PADM_obj(self, mu:float) -> None:
        self.model.setObjective(self.obj["upper_level"] + mu * (self.obj["primal"] - self.obj["dual"]), GRB.MINIMIZE)
    
    def set_valid_ineq_obj(self, i:int) -> None:
        self.model.setObjective(self.vars["upper_level"]["delta"][i], GRB.MAXIMIZE)

    def get_obj_value(self, name:str = None) -> float:
        if name is None:
            return self.model.getObjective().getValue()
        else:
            return self.obj[name].getValue()

    def solve(self) -> None:
        self.model.optimize()
        return self.model.status
    
    def get_values(self, name:str) -> dict[str, list[float]]:
        output = {}
        for key, value in self.vars[name].items():
            if isinstance(value, gp.Var):
                output[key] = value.X
            elif isinstance(value, gp.tupledict):
                output[key] = []
                for index, var in value.items():
                    output[key].append(var.X)
        return output
    
    def get_demands(self):
        return [self.house_params.total_demand *  self.house_params.demands[i] for i in range(self.house_params.hours_num)]
        # self.house_params.total_demand * 
    def get_changed_demands(self):
        return [self.house_params.total_demand * (self.house_params.demands[i] *(1+ self.vars["upper_level"]["delta"][i].X)) for i in range(self.house_params.hours_num)]

class Control():
    def __init__(self, house_params: HouseParams, attack_params: AttackParams) -> None:
        self.house_params = house_params
        self.attack_params = attack_params
    
    @staticmethod
    def diff_values(values_A: dict[str, float | list[float]], values_B: dict[str, float | list[float]]) -> float:
        max_diff = -float('inf')
        for key in values_A:
            if isinstance(values_A[key], int) or isinstance(values_A[key], float):
                list_A = np.array((values_A[key],))
                list_B = np.array((values_B[key],))
            else:
                list_A = np.array(values_A[key])
                list_B = np.array(values_B[key])
            list_diff = max(list_A - list_B)
            max_diff = max(max_diff, list_diff)
        return max_diff

    def primal_model(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_primal_constrs()
        hm.fix_vars("upper_level", 0)
        hm.set_obj("primal")
        hm.solve()
        print(hm.get_values("primal")["capacity_battery"])
        print(hm.get_values("primal")["capacity_PV"])
    
    def dual_model(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_dual_constrs()
        hm.fix_vars("upper_level", 0)
        hm.set_obj("dual")
        hm.solve()
        hm.get_values("dual")
    
    def bigM_attack(self, M):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        hm.add_dual_constrs()
        hm.add_aux_constrs()
        hm.add_bigM_constrs(M)
        hm.set_obj("upper_level")
        hm.solve()
        print(hm.get_obj_value("primal"))
        print(hm.get_obj_value("dual"))
        print(hm.get_values("primal")["capacity_battery"])
        print(hm.get_values("primal")["capacity_PV"])
        return {
            "demands": hm.get_demands(),
            "changed_demands": hm.get_changed_demands(),
            "PV": hm.get_values("primal")["energy_PV"],
            "battery": hm.get_values("primal")["energy_battery"],
            "buy": hm.get_values("primal")["energy_buy"],
            "sell": hm.get_values("primal")["energy_sell"]
        }

    def sos_attack(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        hm.add_dual_constrs()
        hm.add_aux_constrs()
        hm.add_sos_constrs()
        hm.set_obj("upper_level")
        hm.solve()
        print(hm.get_obj_value("primal"))
        print(hm.get_obj_value("dual"))
        return {
            "demands": hm.get_demands(),
            "changed_demands": hm.get_changed_demands(),
            "PV": hm.get_values("primal")["energy_PV"],
            "battery": hm.get_values("primal")["energy_battery"],
            "buy": hm.get_values("primal")["energy_buy"],
        }
    
    def get_ub_valid_ineq(self) -> list[float]:
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.model.Params.LogToConsole = 0
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        ub = []
        for i in range(len(self.house_params.demands)):
            hm.set_valid_ineq_obj(i)
            hm.solve()
            ub.append(hm.get_obj_value())
        return ub

    def sos_valid_ineq_attack(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        hm.add_dual_constrs()
        hm.add_aux_constrs()
        hm.add_sos_constrs()
        ub = self.get_ub_valid_ineq()
        hm.add_valid_ineq_constr(ub)
        hm.solve()
        print(hm.get_obj_value("primal"))
        print(hm.get_obj_value("dual"))
    
    def PADM_attack(self, PADM_params: PADM_Params):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.model.Params.LogToConsole = 0
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        hm.add_dual_constrs()
        hm.set_obj("upper_level")
        hm.solve()
        primal_values = hm.get_values("primal")
        dual_values = hm.get_values("dual")
        upper_level_values = hm.get_values("upper_level")
        mu = PADM_params.initial_mu
        j = 0 # outer loop counter
        while True:
            j += 1
            print(f"j = {j}")
            i = 0 # inner loop counter
            while True:
                i += 1
                print(f"i = {i}")
                hm.set_PADM_obj(mu)
                # fix dual and solve
                hm.fix_vars("dual")
                hm.solve()
                new_primal_values = hm.get_values("primal")
                new_upper_level_values = hm.get_values("upper_level")
                hm.release_vars("dual")
                # fix primal and upper_level and solve
                hm.fix_vars("primal")
                hm.fix_vars("upper_level")
                hm.solve()
                new_dual_values = hm.get_values("dual")
                hm.release_vars("primal")
                hm.release_vars("upper_level")
                # compare the new values with the old ones and the old gets the new
                # if the difference is less than the threshold it is in a stationary point
                # else continue the iterative process
                primal_diff = self.diff_values(primal_values, new_primal_values)
                dual_diff = self.diff_values(dual_values, new_dual_values)
                upper_level_diff = self.diff_values(upper_level_values, new_upper_level_values)
                diff = max(primal_diff, dual_diff, upper_level_diff)
                primal_values = new_primal_values
                dual_values = new_dual_values
                upper_level_values = new_upper_level_values
                if diff < PADM_params.stationary_error or i >= PADM_params.max_stationary_iter:
                    break
            primal_obj_value = hm.get_obj_value("primal")
            dual_obj_value = hm.get_obj_value("dual")
            if abs((primal_obj_value - dual_obj_value)/primal_obj_value) < PADM_params.penalty_error or j >= PADM_params.max_penalty_iter:
                break
            else:
                mu *= PADM_params.increase_factor
        print(hm.get_obj_value("upper_level"))
        print(hm.get_obj_value("primal"))
        print(hm.get_obj_value("dual"))
        print(hm.get_values("primal")["capacity_battery"])
        print(hm.get_values("primal")["capacity_PV"])

            

    




