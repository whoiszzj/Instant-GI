import argparse
import math
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import wandb
from tqdm import tqdm
from generalizable_model.datasets import get_data_loader
from generalizable_model.init_net import InitNet
from generalizable_model.utils import FocalMSELoss, RGBLoss
from utils import loss_fn, compute_psnr, Averager, Timer
from optimizer import Adan


class Trainer:
    def __init__(self, args):
        # load data
        self.train_dataloader = get_data_loader(
            args.dataset.train_dir, args.dataset.batch_size, scale=args.dataset.scale
        )
        self.valid_dataloader = get_data_loader(
            args.dataset.valid_dir, args.dataset.batch_size, scale=args.dataset.scale
        )
        train_len = len(self.train_dataloader)
        valid_len = len(self.valid_dataloader)
        # random select 16 images to visualize
        self.vis_train_indices = sorted(random.sample(range(train_len), 16))
        self.vis_valid_indices = sorted(random.sample(range(valid_len), 16))

        # load model
        self.model = InitNet().cuda()
        if args.train.pretrained:
            checkpoint = torch.load(args.train.pretrained_checkpoint)
            self.model.load_state_dict(checkpoint["model"])
            print(f"Load model from {args.train.pretrained_checkpoint}")

        self.epochs = args.train.epochs
        self.lr_max = args.train.lr
        self.lr_min = args.train.lr_min

        self.pf_loss_fn = FocalMSELoss(alpha=1.0, gamma=2.0, reduction='mean')
        self.rgb_loss_fn = RGBLoss(lambda_val=0.7)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.train.lr)
        # self.optimizer = Adan(self.model.parameters(), lr=args.train.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr_min
        )
        self.checkpoints_dir = None

        # render parameters
        self.background = torch.zeros(3).cuda()
        self.BLOCK_H, self.BLOCK_W = 16, 16
        self.args = args
        self.best_psnr = 0
        self.export_train_gt = True
        self.export_test_gt = True

        if args.train.resume:
            checkpoint = torch.load(args.train.resume_checkpoint)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_psnr = checkpoint["best_psnr"]
            print(f"Resume from {args.train.resume_checkpoint}, start from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0


    def save_epoch(self, epoch_dict):
        epoch_id = epoch_dict["epoch"]
        save_dict = {
            "epoch": epoch_id,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_loss": epoch_dict["train_loss"],
            "best_psnr": self.best_psnr
        }

        if epoch_id % 4 == 0:
            torch.save(save_dict, os.path.join(self.checkpoints_dir, f"epoch_{epoch_id}.pth"))
        # save last epoch: model, optimizer, epoch_id
        torch.save(save_dict, os.path.join(self.checkpoints_dir, "epoch_last.pth"))
        if "test_psnr" not in epoch_dict:
            return
        if epoch_dict["test_psnr"] > self.best_psnr:
            self.best_psnr = epoch_dict["test_psnr"]
            torch.save(save_dict, os.path.join(self.checkpoints_dir, "epoch_best.pth"))

    def backup_code(self):
        backup_list = [
            "generalizable_model",
            "train_init_net.py",
            "train.py",
            "gaussianimage_rs.py"
        ]
        code_dir = os.path.join(self.checkpoints_dir, "code_backup")
        os.makedirs(code_dir, exist_ok=True)
        for file in backup_list:
            os.system(f"cp -r {file} {code_dir}")
        print("Backup code finished.")

    def train(self):
        # init wandb
        wandb.init(
            project="GaussianImage",
            tags=["train_init"], config=self.args
        )
        run_name = wandb.run.name
        # checkpoints save path
        self.checkpoints_dir = args.train.checkpoints_dir + "_" + run_name
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.backup_code()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_dict = self.train_epoch(epoch)
            if epoch_dict is not None:
                self.save_epoch(epoch_dict)
            self.scheduler.step()
            wandb.log({"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)
        print("Training finished.")
        wandb.finish()

    def train_epoch(self, epoch):
        avg_loss = Averager()
        avg_pf_loss = Averager()
        avg_rgb_loss = Averager()
        avg_scaling_loss = Averager()
        avg_psnr = Averager()

        pbar = tqdm(total=len(self.train_dataloader))

        self.model.train()
        train_vis_pred_pfs = []
        train_vis_gt_pfs = []
        train_vis_pred_image = []
        train_vis_gt_image = []
        for i, (gt_image, gt_pf) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            assert gt_image.shape[0] == 1  # batch size must be 1
            gt_image = gt_image.cuda()

            pred_pf, render_img, scaling = self.model(gt_image)
            gt_pf = gt_pf.cuda()

            pf_loss = self.pf_loss_fn(pred_pf, gt_pf)

            rgb_loss_wight = torch.clamp(1. / (pred_pf.detach() + 0.01), 1., 10.)
            rgb_loss = self.rgb_loss_fn(render_img, gt_image, rgb_loss_wight)
            scaling_loss = 10 * torch.relu(0.5 - scaling).mean()

            loss = pf_loss + rgb_loss + scaling_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            with torch.no_grad():
                avg_loss.add(loss.item())
                avg_pf_loss.add(pf_loss.item())
                avg_rgb_loss.add(rgb_loss.item())
                psnr = compute_psnr(render_img, gt_image)
                avg_psnr.add(psnr.item())
                avg_scaling_loss.add(scaling_loss.item())

                if i % 10 == 0 or i == len(self.train_dataloader) - 1:
                    pbar.set_postfix_str(
                        f"Epoch {epoch}, Loss {avg_loss.item():.4f}, "
                        f"lr {self.optimizer.param_groups[0]['lr']:.6f}, "
                        f"pf_loss {avg_pf_loss.item():.4f}, "
                        f"rgb_loss {avg_rgb_loss.item():.4f}, "
                        f"scaling_loss {avg_scaling_loss.item():.4f}, "
                        f"psnr {avg_psnr.item():.4f}"
                    )
                    pbar.update(10)
                if i in self.vis_train_indices:
                    train_vis_pred_pfs.append(pred_pf)
                    train_vis_pred_image.append(render_img.clamp(0, 1))
                    if self.export_train_gt:
                        train_vis_gt_pfs.append(gt_pf)
                        train_vis_gt_image.append(gt_image)
            torch.cuda.empty_cache()
        pbar.close()

        wandb.log({"train/pred_pf": [wandb.Image(img.clamp(0, 1)) for img in train_vis_pred_pfs]}, step=epoch)
        wandb.log({"train/render_image": [wandb.Image(img.clamp(0, 1)) for img in train_vis_pred_image]}, step=epoch)
        if self.export_train_gt:
            wandb.log({"train/gt": [wandb.Image(img.clamp(0, 1)) for img in train_vis_gt_pfs]}, step=epoch)
            wandb.log({"train/gt_image": [wandb.Image(img.clamp(0, 1)) for img in train_vis_gt_image]}, step=epoch)
            self.export_train_gt = False

        pbar.close()
        wandb.log(
            {"train/loss": avg_loss.item(), "train/pf_loss": avg_pf_loss.item(),
             "train/scaling_loss": avg_scaling_loss.item(),
             "train/rgb_loss": avg_rgb_loss.item(), "train/psnr": avg_psnr.item()},
            step=epoch
        )
        print(
            f"Epoch {epoch}, Loss: {avg_loss.item():.4f}, PF Loss: {avg_pf_loss.item():.4f}, "
            f"RGB Loss: {avg_rgb_loss.item():.4f}, PSNR: {avg_psnr.item():.4f}"
        )
        if epoch % 2 == 0:
            test_dict = self.test(epoch)
            return {
                "epoch": epoch,
                "train_loss": avg_loss.item(), "train_psnr": avg_psnr.item(),
                "test_loss": test_dict["test_loss"], "test_psnr": test_dict["test_psnr"]
            }
        else:
            return {
                "epoch": epoch,
                "train_loss": avg_loss.item(), "train_psnr": avg_psnr.item()
            }

    @torch.no_grad()
    def test(self, epoch=0):
        avg_loss = Averager()
        avg_pf_loss = Averager()
        avg_rgb_loss = Averager()
        avg_psnr = Averager()
        avg_scaling_loss = Averager()
        self.model.eval()
        test_vis_pred_pfs = []
        test_vis_gt_pfs = []
        test_vis_pred_image = []
        test_vis_gt_image = []

        pbar = tqdm(total=len(self.valid_dataloader))

        for i, (gt_image, gt_pf) in enumerate(self.valid_dataloader):
            assert gt_image.shape[0] == 1  # batch size must be 1
            gt_image = gt_image.cuda()
            pred_pf, render_img, scaling = self.model(gt_image)
            gt_pf = gt_pf.cuda()
            pf_loss = self.pf_loss_fn(pred_pf, gt_pf)
            rgb_loss_wight = torch.clamp(1. / pred_pf.detach(), 1., 10.)
            rgb_loss = self.rgb_loss_fn(render_img, gt_image, rgb_loss_wight)
            scaling_loss = 10 * torch.relu(0.5 - scaling).mean()
            loss = pf_loss + rgb_loss + scaling_loss
            avg_loss.add(loss.item())
            avg_pf_loss.add(pf_loss.item())
            avg_rgb_loss.add(rgb_loss.item())
            psnr = compute_psnr(render_img, gt_image)
            avg_psnr.add(psnr.item())
            avg_scaling_loss.add(scaling_loss.item())
            if i % 10 == 0:
                pbar.set_postfix_str(f"loss {avg_loss.item():.4f}, psnr {avg_psnr.item():.4f}")
                pbar.update(10)
            if i in self.vis_valid_indices:
                test_vis_pred_pfs.append(pred_pf)
                test_vis_pred_image.append(render_img.clamp(0, 1))
                if self.export_test_gt:
                    test_vis_gt_pfs.append(gt_pf)
                    test_vis_gt_image.append(gt_image)
            torch.cuda.empty_cache()
        pbar.close()
        wandb.log({"test/pred_pf": [wandb.Image(img.clamp(0, 1)) for img in test_vis_pred_pfs]}, step=epoch)
        wandb.log({"test/render_image": [wandb.Image(img.clamp(0, 1)) for img in test_vis_pred_image]}, step=epoch)
        if self.export_test_gt:
            wandb.log({"test/gt": [wandb.Image(img.clamp(0, 1)) for img in test_vis_gt_pfs]}, step=epoch)
            wandb.log({"test/gt_image": [wandb.Image(img.clamp(0, 1)) for img in test_vis_gt_image]}, step=epoch)
            self.export_test_gt = False

        pbar.close()
        wandb.log(
            {"test/loss": avg_loss.item(), "test/pf_loss": avg_pf_loss.item(),
             "test/scaling_loss": avg_scaling_loss.item(),
             "test/rgb_loss": avg_rgb_loss.item(), "test/psnr": avg_psnr.item()},
            step=epoch
        )
        print(
            f"Test Loss: {avg_loss.item():.4f}, PF Loss: {avg_pf_loss.item():.4f}, "
            f"RGB Loss: {avg_rgb_loss.item():.4f}, PSNR: {avg_psnr.item():.4f}"
        )
        return {
            "test_loss": avg_loss.item(), "test_psnr": avg_psnr.item()
        }


def random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--config", type=str, default='./datasets/div2k_train_init_net.yaml', help="Training or testing config"
    )
    args = parser.parse_args(argv)
    with open(args.config, "r") as f:
        args = yaml.safe_load(f)

    def dict_to_namespace(d):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
        })

    return dict_to_namespace(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    random_seed(args.train.seed)

    trainer = Trainer(args)
    trainer.train()
