import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import matplotlib.gridspec as gridspec

def visualize_results(data_dir, init_method):
    """
    Visualize training logs and images from a given data directory
    Args:
        data_dir: Path to data directory containing log.npy, gt.png, init.png and render.png
    """
    # Load log data
    log_path = os.path.join(data_dir, "log.npy")
    log = np.load(log_path)

    # Load gt_pf data
    gt_pf_path = os.path.join(data_dir, "gt_pf.npy")
    gt_pf = np.load(gt_pf_path)

    # Load images
    gt_path = os.path.join(data_dir, "gt.png")
    init_path = os.path.join(data_dir, "init.png")
    render_path = os.path.join(data_dir, "render.png")
    gt_img = np.array(Image.open(gt_path))
    init_img = np.array(Image.open(init_path))
    render_img = np.array(Image.open(render_path))

    # Create figure with 2 rows and 3 columns, all cells等高
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.manager.set_window_title(init_method.upper())  # 设置窗口标题
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Plot loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(log[:, 0], log[:, 1])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    # Plot PSNR
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(log[:, 0], log[:, 2])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR')
    ax2.set_title('PSNR')
    ax2.grid(True)
    
    # Plot gt_pf
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gt_pf, cmap='jet')
    ax3.set_title('Ground Truth PF')
    ax3.axis('off')
    
    # Plot GT image
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(gt_img)
    ax4.set_title('Ground Truth')
    ax4.axis('off')

    # Plot init image
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(init_img)
    ax5.set_title('Initial Result')
    ax5.axis('off')

    # Plot rendered image
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(render_img)
    ax6.set_title('Final Result')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./output/general/0829x2",
                        help='Directory containing log.npy, gt.png, init.png and render.png')
    args = parser.parse_args()
    init_method = os.listdir(args.data_dir)
    for method in init_method:
        visualize_results(os.path.join(args.data_dir, method), method)