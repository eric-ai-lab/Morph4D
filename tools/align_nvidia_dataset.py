# the base is from robust-dynrf, which they only provide the 12 training views and cameras; they also provide a standalone eval script; which use the first view different time as testing

# ! in fact this is not necessary, the test of Ro-Dy is just use the first camera!! we can do in our colmap free manner!

# we have to check the alignment of this camera pose with the original nsff nvidia long; and get the test camera parameters

import os, os.path as osp
import numpy as np

robust_dynrf_root = "../data/nvidia_robust_dynrf/Playground/"
original_nvidia_long_root = "../data/nvidia_nsff_original/Playground"

robust_dynerf_cameras_fn = osp.join(robust_dynrf_root, "poses_bounds.npy")
original_nvidia_long_cameras_fn = osp.join(
    original_nvidia_long_root, "dense", "poses_bounds_cvd.npy"
)

# r_data = np.load(robust_dynerf_cameras_fn)[:, :15]
# o_data = np.load(original_nvidia_long_cameras_fn)[:, :15]


def load_pose(fn):
    poses_bounds = np.load(fn)  # (N_images, 17)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    return poses, H, W, focal

r_pose, r_H, r_W, r_focal = load_pose(robust_dynerf_cameras_fn)
o_pose, o_H, o_W, o_focal = load_pose(original_nvidia_long_cameras_fn)

print("r_pose", r_pose.shape, r_H, r_W, r_focal)
print("o_pose", o_pose.shape, o_H, o_W, o_focal)

for i, r in enumerate(r_pose):
    error = ((o_pose - r)**2).sum(-1).sum(-1)
    print(error.min())


print()


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    """Function for loading LLFF data."""
    poses_arr = np.load(os.path.join(basedir, "poses_bounds_cvd.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = imageio.imread(img0).shape

    sfx = ""

    if factor is not None and factor != 1:
        sfx = "_{}".format(factor)
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(round(sh[1] / factor))
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(round(sh[0] / factor))
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    print("imgdir ", imgdir, " factor ", factor)

    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]

    if poses.shape[-1] != len(imgfiles):
        print(
            "{}: Mismatch between imgs {} and poses {} !!!!".format(
                basedir, len(imgfiles), poses.shape[-1]
            )
        )
        raise NotImplementedError

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :]  # * 1. / factor

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    if not load_imgs:
        imgs = None
    else:
        imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
        imgs = np.stack(imgs, -1)
        print("Loaded image data", imgs.shape, poses[:, -1, 0])

    return poses, bds, imgs, imgfiles
