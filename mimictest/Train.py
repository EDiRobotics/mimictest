import json
from time import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

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
    writer,
    save_path,
    load_epoch_id,
    save_interval,
    step, 
    print_interval,
    bs_per_gpu,
):
    '''
    prof = profile(
        schedule = torch.profiler.schedule(
            wait=20,
            warmup=3,
            active=4,
            repeat=1,
        ),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(save_path+'prof'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
    prof.start()
    '''

    dataset_len = len(prefetcher.loader.dataset)
    avg_reward = 0.0
    for epoch in range(num_training_epochs):
        if epoch % save_interval == 0:
            if epoch != 0: 
                # in the 1st epoch, policy.ema has not been initialized. You may also load the wrong ckpt and modify the right one
                acc.wait_for_everyone()
                acc.save(acc.unwrap_model(policy.net).state_dict(), save_path+f'policy_{epoch+load_epoch_id}.pth')
                policy.ema.copy_to(policy.ema_net.parameters())
                acc.save(acc.unwrap_model(policy.ema_net).state_dict(), save_path+f'ema_{epoch+load_epoch_id}.pth')

                avg_reward = torch.tensor(eva.evaluate_on_env(
                    policy, 
                    num_eval_ep, 
                    max_test_ep_len,
                )).to(device)
                avg_reward = acc.gather_for_metrics(avg_reward).mean(dim=0)

        cum_recon_loss = torch.tensor(0).float().to(device)
        cum_load_time = 0 
        clock = time()
        batch_idx = 0
        batch, load_time = prefetcher.next()
        while batch is not None:
            with acc.accumulate(policy.net):
                policy.net.train()
                optimizer.zero_grad()
                rgbs = preprocessor.rgb_process(batch['rgbs'], train=True)
                low_dims = preprocessor.low_dim_normalize(batch['low_dims'])
                actions = preprocessor.action_normalize(batch['actions'])
                recon_loss = policy.compute_loss(rgbs, low_dims, actions)
                acc.backward(recon_loss)
                optimizer.step(optimizer)
                policy.update_ema() # TODO
                cum_recon_loss += recon_loss.detach() / print_interval
                cum_load_time += load_time / print_interval

            if batch_idx % print_interval == 0 and batch_idx != 0:
                load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(device)
                fps = (bs_per_gpu*print_interval) / (time()-clock)

                avg_recon_loss = acc.gather_for_metrics(cum_recon_loss).mean()
                avg_load_pecnt = acc.gather_for_metrics(load_pecnt).mean()
                fps = acc.gather_for_metrics(torch.tensor(fps).to(device)).sum()

                cum_recon_loss = torch.tensor(0).float().to(device)
                cum_load_time = 0
                clock = time()
                acc.print('Train Epoch: {} [{}/{} ({:.0f}%)] Recon Loss: {:.5f} Reward: {:.5f} FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
                    epoch, 
                    batch_idx * bs_per_gpu * acc.num_processes, 
                    dataset_len, 
                    100. * batch_idx * bs_per_gpu * acc.num_processes / dataset_len, 
                    avg_recon_loss, 
                    avg_reward,
                    fps,
                    avg_load_pecnt,
                    scheduler.get_last_lr()[0],
                ))
                if acc.is_main_process:
                    writer.add_scalar("recon loss", avg_recon_loss, step)
                    writer.add_scalar("reward", avg_reward, step)
                    writer.add_scalar("learning rate", scheduler.get_last_lr()[0], step)
                    writer.add_scalar("FPS", fps, step)
                    writer.add_scalar("loading time in total time", avg_load_pecnt, step)
                    with open(save_path+'step.json', 'w') as json_file:
                        json.dump(step, json_file)
            batch_idx += 1
            step += 1
            batch, load_time = prefetcher.next()
            '''
            prof.step()
            if batch_idx == 28:
                prof.stop()
            '''
        scheduler.step()
