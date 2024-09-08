from time import time
import torch

class DataPrefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        with torch.cuda.stream(self.stream):
            for key in self.batch:
                self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time