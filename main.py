import numpy as np
from model import HouseParams, AttackParams, PADM_Params
from model import Control
import matplotlib.pyplot as plt


PV_availabilities = np.loadtxt("time_series/TS_PVAvail.csv")
demands = np.loadtxt("time_series/TS_Demand.csv")
PV_availabilities = PV_availabilities[0:24]
demands = demands[0:24]

house_params = HouseParams(
    life_time=12*10*30*4,
    price_PV = 1000,
    price_battery = 140,
    cost_buy = 0.25,
    sell_price = 0.05,
    total_demand = 3500,
    demands= demands,
    PV_availabilities= PV_availabilities
)

ub = 0.8
lb = -0.8
attack_params = AttackParams(ub=ub,lb=lb,capacity_battery=1)
PADM_params = PADM_Params()

control = Control(house_params, attack_params)
print(50*"-")
control.primal_model()
print(50*"-")
# control.dual_model()
print(50*"-")
# output = control.bigM_attack(100000)
# print(50*"-")
output = control.sos_attack()
# print(50*"-")
# control.sos_valid_ineq_attack()
# control.PADM_attack(PADM_params)


# prepare data for visualization
demands = np.array(output["demands"])
changed_demands = np.array(output["changed_demands"])
demands_ub = demands * (1+ub)
demands_lb = demands * (1+lb)



# plot
f = plt.figure(1)
plt.subplot(4,1,1)
x = range(len(demands))
plt.plot( x, demands, color="blue", alpha=0.2, label="demands")
# plt.plot(x, demands, color="blue", alpha=0.2)
plt.plot( x, changed_demands, color="red", alpha=0.2, label="changed demands")
plt.plot(x, demands_ub, color="green", linestyle="dashed", alpha=0.1)
plt.plot(x, demands_lb, color="green", linestyle="dashed", alpha=0.1)
plt.ylabel("Demand(kWh)")
plt.title("Demand Curve")
plt.subplot(4,1,2)
plt.plot(x, output["PV"], color="blue", alpha=0.2)
plt.ylabel("PV(kWh)")
plt.subplot(4,1,3)
plt.plot(x, output["battery"], color="blue", alpha=0.2)
plt.ylabel("Battery(kWh)")
plt.subplot(4,1,4)
plt.plot(x, output["buy"], color="blue", alpha=0.2)
plt.ylabel("Buy(kWh)")


g = plt.figure(2)
x = range(len(demands))
plt.plot( x, demands, color="blue", alpha=0.2, label="demands")
# plt.plot(x, demands, color="blue", alpha=0.2)
plt.plot( x, changed_demands, color="red", alpha=0.2, label="changed demands")
plt.plot(x, demands_ub, color="green", linestyle="dashed", alpha=0.1)
plt.plot(x, demands_lb, color="green", linestyle="dashed", alpha=0.1)
plt.ylabel("Demand(kWh)")
plt.title("Demand Curve")
plt.legend()
plt.show()