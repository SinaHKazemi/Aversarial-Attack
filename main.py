from model import HouseParameters, AttackParams, HouseModel

pr = HouseParameters(
    10,
    1,
    1,
    1,
    1,
    1,
    [1,2],
    [3,4]
)   
ap = AttackParams()

hm = HouseModel(pr, ap)
hm._add_vars()
hm._fix_vars("primal")
