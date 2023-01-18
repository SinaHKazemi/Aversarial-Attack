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

attack_params = AttackParams(ub=.8,lb=-.8,capacity_battery=.3)
PADM_params = PADM_Params()

control = Control(house_params, attack_params)
print(50*"-")
control.primal_model()
print(50*"-")
# control.dual_model()
print(50*"-")
output = control.bigM_attack(1000000)
# print(50*"-")
# control.sos_attack()
# print(50*"-")
# control.sos_valid_ineq_attack()
# control.PADM_attack(PADM_params)


# Make the same graph
demands = output["demands"]
changed_demands = output["changed_demands"]
print(demands)
print(changed_demands)
x = range(len(demands))
plt.fill_between( x, demands, color="blue", alpha=0.2, label="label1")
plt.plot(x, demands, color="blue")
plt.fill_between( x, changed_demands, color="red", alpha=0.2, label="label2")
plt.plot(x, changed_demands, color="red")
plt.xlabel("hour")
plt.ylabel("demand(kWh)")
# plt.ylim([0,10])
# plt.xticks(range(5))
plt.title("Demand Curve")
plt.legend()
plt.show()