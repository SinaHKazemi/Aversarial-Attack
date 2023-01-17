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
        self.obj = {} # to keep objective function expressions
        self.constrs = {}
        self.constrs["fix"] = {} # to keep the variable fixing constraints

    def add_vars(self) -> None:
        # upper level
        self.vars["upper_level"] = {}
        self.vars["upper_level"]["delta"] = self.model.addVars(self.house_params.hours_num, vtype= GRB.CONTINUOUS, name="delta")
        
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
        self.vars["aux"]["slack_capacity_PV"] = self.model.addVar(name="slack_limit_PV")
        self.vars["aux"]["slack_capacity_battery"] = self.model.addVar(name="slack_limit_battery")

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

        self.model.update()
        
    def add_upper_level_constrs(self) -> None:
        # add constrainst
        # add objective
        pass
    
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
            self.house_params.cost_PV * self.vars["capacity_PV"] +
            self.house_params.cost_battery * self.vars["capacity_battery"] +
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
        self.obj["dual"] = sum(self.vars["dual"]["eq_demand"] * self.house_params.demands[i] * self.house_params.total_demand * (1 + self.vars["delta"][i]) for i in range(self.house_params.hours_num))

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

        self.constrs["aux"]["slack_capacity_battery"] = self.model.addConstrs(
            (
                sum(self.vars["dual"]["limit_battery"][i] for i in range(self.house_params.hours_num)) +
                self.house_params.cost_battery == 
                self.vars["aux"]["slack_capacity_battery"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_capacity_battery"
        )

        self.constrs["aux"]["slack_capacity_PV"] = self.model.addConstrs(
            (
                sum(self.vars["dual"]["limit_PV"][i] * self.house_params.PV_availabilities[i] for i in range(self.house_params.hours_num)) +
                self.house_params.cost_PV == 
                self.vars["aux"]["slack_capacity_PV"][i] for i in range(self.house_params.hours_num)
            ),
            name="slack_capacity_PV"
        )

    def add_bigM_constrs(self, M:float) -> None:
        self.constrs["bigM"] = {}

        self.constrs["bigM"]["limit_PV"] = {}
        self.constrs["bigM"]["limit_PV"]["var"] = self.model.addConstrs(
            (self.vars["aux"]["limit_PV"] <= self.vars["cs"]["limit_PV"] * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_PV_var"
        )
        self.constrs["bigM"]["limit_PV"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_limit_PV"] <= (1 - self.vars["cs"]["limit_PV"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_PV_con"
        )

        self.constrs["bigM"]["limit_battery"] = {}
        self.constrs["bigM"]["limit_battery"]["var"] = self.model.addConstrs(
            (self.vars["aux"]["limit_battery"] <= self.vars["cs"]["limit_battery"] * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_battery_var"
        )
        self.constrs["bigM"]["limit_battery"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_limit_battery"] <= (1 - self.vars["cs"]["limit_battery"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_limit_battery_con"
        )

        self.constrs["bigM"]["energy_buy"] = {}
        self.constrs["bigM"]["energy_buy"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_buy"] <= self.vars["cs"]["energy_buy"] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_buy_var"
        )
        self.constrs["bigM"]["energy_buy"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_buy"] <= (1 - self.vars["cs"]["energy_buy"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_buy_con"
        )

        self.constrs["bigM"]["energy_sell"] = {}
        self.constrs["bigM"]["energy_sell"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_sell"] <= self.vars["cs"]["energy_sell"] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_sell_var"
        )
        self.constrs["bigM"]["energy_sell"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_sell"] <= (1 - self.vars["cs"]["energy_sell"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_sell_con"
        )

        self.constrs["bigM"]["energy_battery_out"] = {}
        self.constrs["bigM"]["energy_battery_out"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_battery_out"] <= self.vars["cs"]["energy_battery_out"] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_out_var"
        )
        self.constrs["bigM"]["energy_battery_out"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_battery_out"] <= (1 - self.vars["cs"]["energy_battery_out"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_out_con"
        )
        
        self.constrs["bigM"]["energy_battery_in"] = {}
        self.constrs["bigM"]["energy_battery_in"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_battery_in"] <= self.vars["cs"]["energy_battery_in"] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_in_var"
        )
        self.constrs["bigM"]["energy_battery_in"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_battery_in"] <= (1 - self.vars["cs"]["energy_battery_in"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_in_con"
        )

        self.constrs["bigM"]["energy_battery"] = {}
        self.constrs["bigM"]["energy_battery"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_battery"] <= self.vars["cs"]["energy_battery"] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_var"
        )
        self.constrs["bigM"]["energy_battery"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_battery"] <= (1 - self.vars["cs"]["energy_battery"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_battery_con"
        )

        self.constrs["bigM"]["energy_PV"] = {}
        self.constrs["bigM"]["energy_PV"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["energy_PV"] <= self.vars["cs"]["energy_PV"] * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_PV_var"
        )
        self.constrs["bigM"]["energy_PV"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_energy_PV"] <= (1 - self.vars["cs"]["energy_PV"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_energy_PV_con"
        )

        self.constrs["bigM"]["capacity_battery"] = {}
        self.constrs["bigM"]["capacity_battery"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["capacity_battery"] <= self.vars["cs"]["capacity_battery"] * M for i in range(self.house_params.hours_num)),
            name="bigM_capacity_battery_var"
        )
        self.constrs["bigM"]["capacity_battery"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_capacity_battery"] <= (1 - self.vars["cs"]["capacity_battery"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_capacity_battery_con"
        )

        self.constrs["bigM"]["capacity_PV"] = {}
        self.constrs["bigM"]["capacity_PV"]["var"] = self.model.addConstrs(
            (self.vars["primal"]["capacity_PV"] <= self.vars["cs"]["capacity_PV"] * M for i in range(self.house_params.hours_num)),
            name="bigM_capacity_PV_var"
        )
        self.constrs["bigM"]["capacity_PV"]["con"] = self.model.addConstrs(
            (self.vars["aux"]["slack_capacity_PV"] <= (1 - self.vars["cs"]["capacity_PV"]) * M for i in range(self.house_params.hours_num)),
            name="bigM_capacity_PV_con"
        )
        
    def add_sos_constrs(self) -> None:
        self.constrs["sos"] = {}
        self.constrs["sos"]["limit_PV"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["aux"]["limit_PV"],self.vars["aux"]["slack_limit_PV"]])
        self.constrs["sos"]["limit_battery"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["aux"]["limit_battery"],self.vars["aux"]["slack_limit_battery"]])
        self.constrs["sos"]["energy_buy"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_buy"],self.vars["aux"]["slack_energy_buy"]])
        self.constrs["sos"]["energy_sell"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_sell"],self.vars["aux"]["slack_energy_sell"]])
        self.constrs["sos"]["energy_battery_out"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_battery_out"],self.vars["aux"]["slack_energy_battery_out"]])
        self.constrs["sos"]["energy_battery_in"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_battery_in"],self.vars["aux"]["slack_energy_battery_in"]])
        self.constrs["sos"]["energy_battery"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_battery"],self.vars["aux"]["slack_energy_battery"]])
        self.constrs["sos"]["energy_PV"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["energy_PV"],self.vars["aux"]["slack_energy_PV"]])
        self.constrs["sos"]["capacity_battery"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["capacity_battery"],self.vars["aux"]["slack_capacity_battery"]])
        self.constrs["sos"]["capacity_PV"] = self.model.addSOS(GRB.SOS_TYPE1, [self.vars["primal"]["capacity_PV"],self.vars["aux"]["slack_capacity_PV"]])

        # https://www.gurobi.com/documentation/9.5/refman/presos1encoding.html#parameter:PreSOS1Encoding
        # to shut off the reformulation PreSOS1BigM parameter should be set to zero
        self.model.setParam("PreSOS1BigM", 0)

    def add_valid_ineq_constr(self, ub:list[float]) -> None:
        self.constrs["valid_ineq"] = self.model.addConstr(
            self.obj["primal"] <= sum(self.vars["dual"]["eq_demand"] * self.house_params.demands[i] * self.house_params.total_demand * (1 + ub[i]) for i in range(self.house_params.hours_num)),
            name="valid_ineq"
        )

    def fix_vars(self, name: str, fix_value: None) -> None:
        var_list = []
        for value in self.vars[name].values():
            if isinstance(value,gp.tupledict):
                var_list.extend(list(value))
            else: # isinstance(value,gp.Var)
                var_list.append(value)
        constr_list = []
        self.constrs["fix"][name] = constr_list
        for var in var_list:
            if fix_value is None:
                constr_list.append(self.model.addConstr(var.X <= var))
                constr_list.append(self.model.addConstr(var.X >= var))
            else:
                constr_list.append(self.model.addConstr(fix_value <= var))
                constr_list.append(self.model.addConstr(fix_value >= var))

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
        for key, value in self.vars[name]:
            if isinstance(value, gp.Var):
                output[key] = value.X
            elif isinstance(value, gp.tupledict):
                output[key] = []
                for var in value:
                    output[key].append(var.X)
        return output

class Control():
    def __init__(self, house_params: HouseParameters, attack_params: AttackParams) -> None:
        self.house_params = house_params
        self.attack_params = attack_params
    
    def primal_model(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_primal_constrs()
        hm.fix_vars("upper_level", 0)
        hm.set_obj("primal")
        hm.solve()
        hm.get_values("primal")
    
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
        hm.set_obj("uppr_level")
        hm.solve()
        # get results

    def sos_attack(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        hm.add_dual_constrs()
        hm.add_aux_constrs()
        hm.add_sos_constrs()
        hm.set_obj("uppr_level")
        hm.solve()
    
    def get_ub_valid_ineq(self) -> list[float]:
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
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
        # calc ub for valid inequality
        hm.add_valid_ineq_constr()
        hm.solve() 
    
    def PADM_attack(self):
        hm = HouseModel(house_params=self.house_params, attack_params=self.attack_params)
        hm.add_vars()
        hm.add_upper_level_constrs()
        hm.add_primal_constrs()
        hm.add_dual_constrs()
        hm.solve()

    




