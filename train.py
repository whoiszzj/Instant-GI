import math
import time
from pathlib import Path
import argparse
from types import SimpleNamespace
import cv2
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from sympy.polys.polyoptions import Gaussian

from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import wandb
from torch_kdtree import build_kd_tree
from generalizable_model.init_net import InitNet
from generalizable_model.utils import dither_image
from quard_image import QuardImage


def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor


@torch.no_grad()
def init_from_net(image, model):
    t_1 = time.time()
    xy, scale, rotation, color, _ = model(image, get_gaussians=True)
    t_2 = time.time()
    print("Init time: ", t_2 - t_1)
    xy = xy.cpu().numpy()
    scale = scale.cpu().numpy()
    rotation = rotation.cpu().numpy()
    color = color.cpu().numpy()
    # tri = tri.cpu().view(-1, 6).numpy()
    init_gaussians = np.concatenate([xy, scale, rotation, color], axis=1)
    return init_gaussians, t_2 - t_1


class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""

    def __init__(
            self,
            image_path: Path,
            num_points=None,
            save_path=None,
            init_net_model=None,
            args=None
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.num_points = num_points
        self.init_points = None

        if args.model.init_gaussians == "net":
            self.init_points, self.init_time = init_from_net(self.gt_image, init_net_model)
            # torch.cuda.empty_cache()
            self.num_points = len(self.init_points)
        elif args.model.init_gaussians == "random":
            if args.model.random_init.same_test:
                sampled_points, _ = init_from_net(self.gt_image, init_net_model)
                # torch.cuda.empty_cache()
                self.num_points = len(sampled_points)
            self.init_time = 0
        elif args.model.init_gaussians == "quard":
            quard_image = QuardImage(image_path)
            self.init_points, self.init_time = quard_image.split()
            self.num_points = len(self.init_points)

        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = args.train.iterations
        self.log_dir = Path(save_path)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logwriter = LogWriter(self.log_dir, train=(args.mode == "train"))
        self.logwriter.write("Image name: {}".format(image_path.stem))
        self.logwriter.write("Image size: {}x{}".format(self.H, self.W))
        self.logwriter.write("Number of points: {}".format(self.num_points))

        if args.model.model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(
                loss_type="L2", opt_type="adan", num_points=self.num_points,
                H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                device=self.device, lr=args.train.lr, quantize=False,
                init_points=self.init_points, gt_image=self.gt_image
            ).to(self.device)
        else:
            raise NotImplementedError

        if args.mode == "test" and args.load_path is not None:
            print(f"loading model path:{args.load_path}")
            checkpoint = torch.load(args.load_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
        
        self.use_wandb = args.wandb.activate
        self.record_iter = args.wandb.record_iter

    def data_normalize(self, data):
        data_mean = data.mean()
        data_std = data.std()
        data_max = min(data_mean + 3 * data_std, data.max())
        data_min = max(data_mean - 3 * data_std, data.min())
        data = (data - data_min) / (data_max - data_min)
        data = torch.clamp(data, 0, 1)
        return data

    def get_gt_pf(self):
        xy = self.gaussian_model.get_xyz.detach()
        # xy to pixel
        xy[..., 0] = (xy[..., 0] + 1) * self.W / 2
        xy[..., 1] = (xy[..., 1] + 1) * self.H / 2
        # build kd tree
        kd_tree = build_kd_tree(xy)
        # xy map
        xy_map = torch.meshgrid(
            torch.arange(self.W, device=self.device),
            torch.arange(self.H, device=self.device),
            indexing='xy'
        )
        xy_map = torch.stack(xy_map, dim=-1).reshape(-1, 2).float()  # [H*W, 2]
        # query
        k = 10
        dist, idx = kd_tree.query(xy_map, nr_nns_searches=k)
        dist = torch.sqrt(dist)
        dist_max = dist.max(dim=-1)[0]  # [H*W, 1]
        density = k / torch.sqrt(dist_max)
        density = self.data_normalize(density)
        # density = (density - density.min()) / (density.max() - density.min())
        density = density.view(self.H, self.W)
        # dither_image(density.unsqueeze(0))

        gt_pf = density.cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(gt_pf)
        # xy_np = xy.cpu().numpy()
        # plt.scatter(xy_np[:, 0], xy_np[:, 1], c='r', s=1)
        # plt.colorbar()
        # plt.show()
        return gt_pf

    def train(self):
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training progress")
        self.gaussian_model.train()
        start_time = time.time() - self.init_time
        log_np = []
        with torch.no_grad():
            init_output = self.gaussian_model()
            init_img = (init_output["render"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(self.log_dir / "init.png"), cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB))

        for iteration in range(1, self.iterations + 1):
            loss, psnr, image = self.gaussian_model.train_iter(self.gt_image)
            with torch.no_grad():
                if iteration % 200 == 1 or iteration == self.iterations:
                    progress_bar.set_postfix({f"Loss": f"{loss.item():.{7}f}", "PSNR": f"{psnr:.{4}f},"})
                    progress_bar.update(200)
                    if self.use_wandb:
                        wandb.log({"Loss": loss.item(), "PSNR": psnr}, step=iteration)
                    log_np.append([iteration, loss.item(), psnr])
                if self.use_wandb and (iteration % self.record_iter == 1 or iteration == self.iterations):
                    img = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    wandb.log({"Image": [wandb.Image(img, caption=f"iter:{iteration}")]}, step=iteration)

        end_time = time.time() - start_time
        progress_bar.close()

        # save log np
        log_np = np.array(log_np).astype(np.float32)
        np.save(self.log_dir / "log.npy", log_np)

        gt_pf = self.get_gt_pf()
        np.save(self.log_dir / "gt_pf.npy", gt_pf)

        psnr_value, ms_ssim_value, num_points, fps = self.test()

        self.logwriter.write(
            "Training Complete in {:.4f}s, FPS:{:.4f}".format(end_time, fps)
        )
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        return psnr_value, ms_ssim_value, self.num_points, end_time, fps

    def limit_train(self, log_times):
        self.gaussian_model.train()
        img_log = []
        psnr_log = []
        time_logged = {t: False for t in log_times}

        start_time = time.time() - self.init_time

        for iteration in range(1, self.iterations + 1):
            loss, psnr, image = self.gaussian_model.train_iter(self.gt_image)
            with torch.no_grad():
                cur_time = time.time() - start_time
                if iteration == 1:
                    img = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_log.append(img)
                    psnr_log.append(psnr)
                if iteration % 20 == 1:
                    stop = False
                    for t in log_times:
                        if cur_time > t and not time_logged[t]:
                            print(f"Time {t}s, psnr: {psnr}, cur_time: {cur_time}")
                            time_logged[t] = True
                            img = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            img_log.append(img)
                            psnr_log.append(psnr)
                            if t == log_times[-1]:
                                stop = True
                                print(f"Stop training at time {t} s")
                            break
                    if stop:
                        break

            if self.use_wandb:
                for i in range(len(img_log)):
                    img = img_log[i]
                    psnr = psnr_log[i]
                    wandb.log({"Image": wandb.Image(img), "PSNR": psnr}, step=i)

        end_time = time.time() - start_time
        psnr_value, ms_ssim_value, num_points, fps = self.test()

        self.logwriter.write(
            "Training Complete in {:.4f}s, FPS:{:.4f}".format(end_time, fps)
        )
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        return psnr_value, ms_ssim_value, self.num_points, end_time, fps

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        # test fps
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time) / 100
        # save rendered image
        img = (out["render"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(self.log_dir / "render.png"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gt_img = (self.gt_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(self.log_dir / "gt.png"), cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        return psnr, ms_ssim_value, self.init_points, 1 / test_end_time


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--config", type=str, default='./datasets/kodak.yaml', help="Training dataset"
    )
    args = parser.parse_args(argv)
    with open(args.config, "r") as f:
        args = yaml.safe_load(f)

    def dict_to_namespace(d):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()
        })

    return dict_to_namespace(args)


def random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


def main(argv):
    
    # 1. load parameters
    args = parse_args(argv)
    num_point = args.model.random_init.num_points
    init_net_model = None
    random_seed(args.train.seed)
    logwriter = LogWriter(Path(args.train.save_path), train=(args.mode == "train"))
    psnrs, ms_ssims, num_points, training_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    data_name = args.dataset.dataset_name

    # 2. load init_net
    if args.mode == "train":
        if args.model.init_gaussians == "net" or (args.model.init_gaussians == "random" and args.model.random_init.same_test):
            init_net_model = InitNet(kernel_size=args.model.kernel_size).cuda()
            init_net_model.load_state_dict(torch.load(args.model.init_model_path)["model"])

    # 3. load image paths
    image_paths = []
    if args.dataset.dataset_name != "general":
        for i in range(args.dataset.start, args.dataset.start + args.dataset.images_num):
            image_paths.append(Path(args.dataset.images_dir) / args.dataset.regex.format(i))
    else:
        # check image's existence
        for image_path in args.dataset.image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found")
            image_paths.append(Path(image_path))
    print(f"There are {len(image_paths)} images to {args.mode}")
    if len(image_paths) == 0:
        raise ValueError("No images found")

    # 4. train or test
    for image_path in image_paths:
        image_name = data_name + "/" + image_path.stem
        trainer = SimpleTrainer2d(
            image_path=image_path, 
            save_path=os.path.join(args.train.save_path, image_name, args.model.init_gaussians),
            num_points=num_point, init_net_model=init_net_model, args=args
        )
        
        if args.mode == "train":
            print(f"Training {image_path}")
            if args.wandb.activate:
                wandb.init(
                    project=args.wandb.project_name, name=image_name, config=args.__dict__,
                    group=data_name,
                    tags=[
                        args.model.init_gaussians,
                        f"iterations:{args.train.iterations}",
                    ]
                )
            if args.train.limit_train:
                psnr, ms_ssim, num_point, training_time, eval_fps = trainer.limit_train(args.train.limit_train_log_point)
            else:
                psnr, ms_ssim, num_point, training_time, eval_fps = trainer.train()
            if args.wandb.activate:
                wandb.finish()
        else:
            print(f"Testing {image_path}")
            psnr, ms_ssim, num_point, eval_fps = trainer.test()
            training_time = 0

        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        num_points.append(num_point)
        training_times.append(training_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write(
            "{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Num:{}, Training:{:.4f}s, FPS:{:.4f}".format(
                image_name, trainer.H, trainer.W, psnr, ms_ssim, num_point, training_time, eval_fps
            )
        )
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_num_points = torch.tensor(num_points).float().mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h // len(image_paths)
    avg_w = image_w // len(image_paths)

    logwriter.write(
        "Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Num:{:.2f} Training:{:.4f}s, FPS:{:.4f}".format(
            avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_num_points, avg_training_time, avg_eval_fps
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
