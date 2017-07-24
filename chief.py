import torch
import torch.optim as optim
import torch.multiprocessing as mp
import time

def chief(rank, params, traffic_light, counter, shared_p, shared_v, optimizer_p, optimizer_v):

    while True:
        time.sleep(1)

        # workers will wait after last loss computation
        if counter.get() > params.update_treshold:
            shared_p.synchronize()
            shared_v.synchronize()
            optimizer_p.step()
            optimizer_v.step()
            counter.reset()
            shared_p.reset_grads()
            shared_v.reset_grads
            traffic_light.switch() # workers start new loss computation
