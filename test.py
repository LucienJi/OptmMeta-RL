from isaacgym import gymapi

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.physx.use_gpu = True
try:
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
except:
    print("except")

print("done")