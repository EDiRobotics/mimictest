import copy
import json
import glob
from time import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import wandb
from tqdm import tqdm

def train(
    acc, 
    prefetcher, 
    preprocessor, 
    policy, 
    optimizer, 
    scheduler, 
    num_training_epochs,
    eva, 
    num_eval_ep, 
    max_test_ep_len,
    device, 
    save_path,
    load_epoch_id,
    save_interval,
    print_interval,
    bs_per_gpu,
    max_grad_norm,
    record_video,
    do_profile,
):
    if do_profile:
        prof = profile(
            schedule = torch.profiler.schedule(
                wait=20,
                warmup=3,
                active=4,
                repeat=1,
            ),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(save_path/'prof'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        prof.start()

    dataset_len = len(prefetcher.loader.dataset)
    avg_reward = 0.0
    for epoch in tqdm(range(num_training_epochs), desc=f"train epochs", disable=not acc.is_main_process):
        if epoch % save_interval == 0:
            if epoch != 0: 
                # in the 1st epoch, policy.ema has not been initialized. You may also load the wrong ckpt and modify the right one
                policy.save_pretrained(acc, save_path, epoch+load_epoch_id)
                avg_reward = torch.tensor(eva.evaluate_on_env(
                    acc,
                    policy, 
                    epoch,
                    num_eval_ep, 
                    max_test_ep_len,
                    record_video,
                )).to(device)
                avg_reward = acc.gather_for_metrics(avg_reward).mean(dim=0)

        batch_metric = {
            'loss': 0,
            'grad_norm': 0,
            'dataload_time': 0,
        } 
        avg_metric = copy.deepcopy(batch_metric) # average over batches
        clock = time()
        batch_idx = 0
        batch, batch_metric['dataload_time'] = prefetcher.next()
        while batch is not None:
            with acc.accumulate(policy.net):
                policy.net.train()
                optimizer.zero_grad()
                rgbs = preprocessor.rgb_process(batch['rgbs'], train=True)
                low_dims = preprocessor.low_dim_normalize(batch['low_dims'])
                actions = preprocessor.action_normalize(batch['actions'])
                loss = policy.compute_loss(rgbs, low_dims, actions)
                acc.backward(loss)
                if acc.sync_gradients:
                    batch_metric['grad_norm'] = acc.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step(optimizer)
                if policy.use_ema and batch_idx % policy.ema_interval == 0:
                    policy.update_ema()
                batch_metric['loss'] = loss.detach()
                for key in batch_metric:
                    avg_metric[key] += batch_metric[key] / print_interval

            if batch_idx % print_interval == 0 and batch_idx != 0:
                avg_metric['dataload_percent_first_gpu'] = avg_metric['dataload_time'] * print_interval / (time()-clock)
                avg_metric['lr'] = scheduler.get_last_lr()[0]
                avg_metric['reward'] = avg_reward
                avg_metric['fps_first_gpu'] = (bs_per_gpu*print_interval) / (time()-clock)
                clock = time()

                for key in batch_metric:
                    if key != 'dataload_time':
                        avg_metric[key] = acc.gather_for_metrics(avg_metric[key]).mean()
                text = '\nTrain Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, 
                    batch_idx * bs_per_gpu * acc.num_processes, 
                    dataset_len, 
                    100. * batch_idx * bs_per_gpu * acc.num_processes / dataset_len, 
                )
                for key in avg_metric:
                    text = text + ' {}: {:.5f}'.format(key, avg_metric[key])
                acc.print(text)
                acc.log(avg_metric)
                for key in avg_metric:
                    avg_metric[key] = 0 

            batch_idx += 1
            batch, batch_metric['dataload_time'] = prefetcher.next()
            if do_profile:
                prof.step()
                if batch_idx == 28:
                    prof.stop()
                    acc.print("Profiling log saved in ", str(save_path/'prof'))
                    acc.print("Visualize the profiling log by tensorboard with torch_tb_profiler plugin, see https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html")
        scheduler.step()
