# train_diffusion_unet_dp3_workspace.py
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import MinkowskiEngine as ME
import wandb
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from termcolor import cprint
from pcdp.workspace.base_workspace import BaseWorkspace
from pcdp.policy.diffusion_RISE_policy import RISEPolicy
from pcdp.dataset.base_dataset import BasePointCloudDataset
from pcdp.dataset.RISE_stack_pc_dataset import collate_fn
from pcdp.env_runner.base_pointcloud_runner import BasePointCloudRunner 
import pcdp.common.mono_time as mono_time
from pcdp.common.checkpoint_util import TopKCheckpointManager
from pcdp.common.pytorch_util import dict_apply, optimizer_to
from pcdp.model.diffusion.ema_model import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainRISEWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg:OmegaConf, output_dir = None):
        super().__init__(cfg, output_dir=output_dir)
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # configure model
        self.model: RISEPolicy = hydra.utils.instantiate(cfg.policy)
        
        self.ema_model: RISEPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
    
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_parameters/1e6:.2f}M")
    
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params = self.model.parameters())
        
        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # device
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {lastest_ckpt_path}", "blue", attrs=["bold"])
                self.load_checkpoint(path=lastest_ckpt_path)

        # dataset
        dataset: BasePointCloudDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BasePointCloudDataset)
        dataloader = DataLoader(dataset, collate_fn=collate_fn, **cfg.dataloader)

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, **cfg.val_dataloader)

        # env runner
        env_runner: BasePointCloudRunner = hydra.utils.instantiate(
            cfg.task.env_runner, 
            output_dir=self.output_dir)

        # optimizer and lr scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps= 2000,
            num_training_steps=len(dataloader) * cfg.training.num_epochs
        )
        if cfg.training.resume and self.global_step > 0:
            lr_scheduler.last_epoch = self.global_step - 1

        # ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            self.ema_model.to(device)

        # logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )

        # training loop
        train_history = []
        if cfg.training.resume:
            history_path = os.path.join(self.output_dir, 'train_history.npy')
            if os.path.exists(history_path):
                train_history = np.load(history_path).tolist()

        start_epoch = self.epoch
        self.model.train()
        for epoch in range(start_epoch, cfg.training.num_epochs):
            self.epoch = epoch
            
            avg_loss_epoch = 0
            pbar = tqdm.tqdm(dataloader, desc=f"Epoch {self.epoch+1}/{cfg.training.num_epochs}")

            for data in pbar:
                # data to device
                cloud_coords = data['input_coords_list'].to(device)
                cloud_feats = data['input_feats_list'].to(device)
                action_data = data['action_normalized'].to(device)
                
                
                # forward
                cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
                loss = self.model(cloud_data, action_data, batch_size=action_data.shape[0])
                
                # backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lr_scheduler.step()

                if cfg.training.use_ema:
                    ema.step(self.model)

                # logging
                loss_val = loss.item()
                avg_loss_epoch += loss_val
                pbar.set_postfix(loss=f"{loss_val:.4f}")
                wandb_run.log({
                    'train_loss_step': loss_val,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=self.global_step)
                self.global_step += 1

            avg_loss_epoch /= len(dataloader)
            train_history.append(avg_loss_epoch)
            print(f"Epoch {self.epoch+1} Train loss: {avg_loss_epoch:.6f}")
            epoch_log = {
                'train_loss_epoch': avg_loss_epoch,
                'epoch': self.epoch
            }
            if (self.epoch + 1) % cfg.training.validation_every == 0 and len(val_dataloader) > 0:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_dataloader:
                        # data to device
                        cloud_coords = data['input_coords_list'].to(device)
                        cloud_feats = data['input_feats_list'].to(device)
                        action_data = data['action_normalized'].to(device)
                        cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
                        loss = self.model(cloud_data, action_data,batch_size=action_data.shape[0])
                        val_loss += loss.item()
                val_loss /= len(val_dataloader)
                epoch_log['val_loss_epoch'] = val_loss
                self.model.train()

            # run rollout
            if (self.epoch + 1) % cfg.training.rollout_every == 0:
                policy_for_rollout = self.ema_model if cfg.training.use_ema else self.model
                runner_log = env_runner.run(policy_for_rollout)
                epoch_log.update(runner_log)

            wandb_run.log(epoch_log, step=self.global_step)

            # checkpointing
            if (self.epoch + 1) % cfg.training.save_epochs == 0:
                ckpt_path = os.path.join(self.output_dir, 'checkpoints', f"policy_epoch_{self.epoch + 1}_seed_{cfg.training.seed}.ckpt")
                self.save_checkpoint(path=ckpt_path)
                self.save_checkpoint()
                
                # plot history
                plt.figure()
                plt.plot(range(len(train_history)), train_history, label='Train Loss')
                plt.tight_layout()
                plt.legend()
                plt.title("Training Loss History")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(os.path.join(self.output_dir, f'train_history_seed_{cfg.training.seed}.png'))
                plt.close()
                np.save(os.path.join(self.output_dir, 'train_history.npy'), np.array(train_history))

        # save final model
        last_ckpt_path = os.path.join(self.output_dir, 'checkpoints', "policy_last.ckpt")
        self.save_checkpoint(path=last_ckpt_path)
        cprint("Training finished!", "green", attrs=["bold"])
        
        






        


        
