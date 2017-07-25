import torch
import torch.optim as optim
import torch.multiprocessing as mp
import time

def chief(rank, params, traffic_light, counter, shared_model, optimizer):

    while True:
        time.sleep(1)

        # workers will wait after last loss computation
        if counter.get() > params.update_treshold:
            shared_model.synchronize()
            optimizer.step()
            counter.reset()
            shared_model.reset_grads()
            traffic_light.switch() # workers start new loss computation
