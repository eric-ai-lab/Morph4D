import time

import torch
import os, os.path as osp
import logging
from lib_4d.solver_gs import render_test, render_test_tto
import numpy as np
from lib_4d.camera import SimpleFovCamerasIndependent
from lib_4d.gs_static_model import StaticGaussian
from lib_4d.gs_ed_model import DynSCFGaussian
from lib_4d.eval_utils.eval_nvidia import eval_nvidia_dir

import imageio
from omegaconf import OmegaConf
from lib_data.iphone_helpers import load_iphone_gt_poses
from lib_data.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test
from lib_data.nerfies_helpers import load_nerfies_gt_poses
from lib_4d.autoencoder.model import load_head

import json
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def test_main(
    cfg,
    saved_dir,
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
):
    # ! this func can be called at the end of running, or run seperately after trained

    # get cfg
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "dataset_mode", "iphone")
    max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

    d_model_ckpt = torch.load(osp.join(saved_dir, "finetune_d_model.pth"))
    d_model = DynSCFGaussian.load_from_ckpt(d_model_ckpt, device=device)
    # * load static model and camera again, because it's also finetuned
    saved_cam = torch.load(osp.join(saved_dir, "finetune_s_model_cam.pth"))
    cams: SimpleFovCamerasIndependent = SimpleFovCamerasIndependent(
        T=len(saved_cam["q_wc"]),
        fovdeg_init=40.0,  # dummy init
    )
    cams.load_state_dict(saved_cam, strict=True)

    s_model = StaticGaussian(
        load_fn=osp.join(saved_dir, "finetune_s_model.pth"),
        max_sph_order=max_sph_order,
    ).to(device)

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    ######################################################################
    ######################################################################

    if dataset_mode == "iphone":
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            _,
            gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(data_root, getattr(cfg, "t_subsample", 1))
        gt_dir = osp.join(data_root, "test_images")
        # * cfg
        tto_viz_interval = 50
        tto_steps = getattr(cfg, "tto_steps", 30)
        decay_start = getattr(cfg, "tto_decay_start", 15)
        lr_p = getattr(cfg, "tto_lr_p", 0.003)
        lr_q = getattr(cfg, "tto_lr_q", 0.003)
        lr_final = getattr(cfg, "tto_lr_final", 0.0001)
        sgd_flag = False
    elif dataset_mode == "nerfies":
        while data_root.endswith("left1") or data_root.endswith("right1") or data_root.endswith("preprocess"):
            data_root = osp.dirname(data_root)

        (
            gt_training_cam_T_wi,
            gt_training_tids,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_nerfies_gt_poses(data_root, getattr(cfg, "t_subsample", 1))
        gt_dir = osp.join(data_root, "right1/images")
        # * cfg
        tto_viz_interval = 50
        tto_steps = getattr(cfg, "tto_steps", 30)
        decay_start = getattr(cfg, "tto_decay_start", 15)
        lr_p = getattr(cfg, "tto_lr_p", 0.003)
        lr_q = getattr(cfg, "tto_lr_q", 0.003)
        lr_final = getattr(cfg, "tto_lr_final", 0.0001)
        sgd_flag = False
    elif dataset_mode == "nvidia":
        # ! always use the first training view
        gt_training_cam_T_wi = cams.T_wc_list().detach().cpu()
        gt_training_fov = cams.fov

        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)
        if os.path.basename(data_root) == "preprocess":
            name = os.path.basename(osp.dirname(data_root))
        else:
            name = os.path.basename(data_root)
        gt_dir = osp.join(
            "./data/nvidia_dev/gt/", name
        )
        # * cfg
        tto_viz_interval = 1
        tto_steps = getattr(cfg, "tto_steps", 100)
        decay_start = getattr(cfg, "tto_decay_start", 30)
        lr_p = getattr(cfg, "tto_lr_p", 0.0003)
        lr_q = getattr(cfg, "tto_lr_q", 0.0003)
        lr_final = getattr(cfg, "tto_lr_final", 0.000001)
        sgd_flag = False

    else:
        raise ValueError(
            f"Unknown dataset mode: {dataset_mode}, shouldn't call test funcs"
        )
    # id the image size
    sample_fn = [
        f for f in os.listdir(gt_dir) if f.endswith(".png") or f.endswith(".jpg")
    ][0]
    sample = imageio.imread(osp.join(gt_dir, sample_fn))
    H, W = sample.shape[:2]

    ######################################################################
    ######################################################################

    eval_prefix = "tto_" if tto_flag else ""

    if not skip_test_gen:
        for test_i in range(len(gt_testing_cam_T_wi_list)):
            testing_fov = gt_testing_fov_list[test_i]
            testing_focal = 1.0 / np.tan(np.deg2rad(testing_fov) / 2.0)

            if tto_flag:
                frames, test_latent_features = render_test_tto(
                    gt_rgb_dir=gt_dir,
                    tto_steps=tto_steps,
                    decay_start=decay_start,
                    lr_p=lr_p,
                    lr_q=lr_q,
                    lr_final=lr_final,
                    use_sgd=sgd_flag,
                    viz_interval=tto_viz_interval,
                    #
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
                    test_camera_tid=gt_testing_tids_list[test_i],
                    save_dir=osp.join(saved_dir, f"tto_test"),
                    save_viz_dir=osp.join(saved_dir, "tto_test_viz"),
                    save_pose_fn=osp.join(saved_dir, f"tto_test_pose_{test_i}"),
                    fn_list=gt_testing_fns_list[test_i],
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[test_i],
                )
                imageio.mimsave(
                    osp.join(saved_dir, f"tto_test_cam{test_i}.mp4"), frames
                )
                torch.save(test_latent_features, osp.join(saved_dir, f"tto_test_latent_features.pth"))
            else:
                frames, test_latent_features = render_test(
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
                    test_camera_tid=gt_testing_tids_list[test_i],
                    save_dir=osp.join(saved_dir, "test"),
                    fn_list=gt_testing_fns_list[test_i],
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[test_i],
                )
                imageio.mimsave(osp.join(saved_dir, f"test_cam{test_i}.mp4"), frames)
                torch.save(test_latent_features, osp.join(saved_dir, f"test_latent_features.pth"))

    # * Test
    if dataset_mode == "iphone":
        from lib_4d.eval_utils.eval_dyncheck import eval_dycheck

        eval_dycheck(
            save_dir=saved_dir,
            gt_rgb_dir=gt_dir,
            gt_mask_dir=osp.join(data_root, "test_covisible"),
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            strict_eval_all_gt_flag=True,  # ! only support full len now!!
            eval_non_masked=eval_also_dyncheck_non_masked,
        )
    elif dataset_mode == "nerfies":
        from lib_4d.eval_utils.eval_nerfies import evaluate as eval_nerfies

        eval_nerfies(
            save_dir=saved_dir,
            gt_dir=gt_dir,
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            gt_testing_fns=gt_testing_fns_list[0]
        )
        latent_features = torch.load(osp.join(saved_dir, f"{eval_prefix}test_latent_features.pth"), weights_only=True, map_location=device)

    elif dataset_mode == "nvidia":
        if data_root.endswith("/"):
            data_root = data_root[:-1]
        eval_nvidia_dir(
            gt_dir=gt_dir,
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            report_dir=osp.join(saved_dir, f"{eval_prefix}test_report"),
        )
        latent_features = torch.load(osp.join(saved_dir, f"tto_test_latent_features.pth"), weights_only=True, map_location=device)

    logging.info(f"Finished, saved to {saved_dir}")

    logging.info(f"decoding latent feature(lseg)")
    feature_config = osp.join(saved_dir, "feature_config.yaml")
    feature_head_ckpt_path = osp.join(saved_dir, "finetune_semantic_heads.pth")
    feature_head = load_head(feature_config=feature_config, feature_head_ckpt_path=feature_head_ckpt_path, device=device)
    if "langseg" in feature_head.keys() :
        langseg_save_dir = osp.join(saved_dir, f"{eval_prefix}test_lseg")
        # if not os.path.exists(langseg_save_dir):
        os.makedirs(langseg_save_dir, exist_ok=True)
        decode_time=0
        for i in range(len(latent_features)):
            start_time = time.time()
            latent_feat = latent_features[i].permute(1,2,0)
            langseg_feat = feature_head.decode("langseg", latent_feat)
            langseg_feat = langseg_feat.permute(2,0,1).cpu()
            decode_time += time.time() - start_time
            torch.save(langseg_feat, osp.join(langseg_save_dir, f"{i:05d}_fmap_CxHxW.pt"))
        print(f"decode time: {decode_time}")
        with open(osp.join(langseg_save_dir, "lseg_decode_time.json"), "w") as f:
            json.dump({"decode_time": decode_time}, f)
    else:
        logging.info(f"no langseg in feature head, skip decoding")
    logging.info(f"decoding latent feature(lseg) finished")


    return


if __name__ == "__main__":

    # test_main(
    #     # saved_dir="./data/iphone_1x_dev/block/log/native_iphone_base.yaml20240507_155311",
    #     # data_root="./data/iphone_1x_dev/block/",
    #     saved_dir="./data/iphone_1x_dev/paper-windmill/log/native_iphone_base.yaml20240507_175337",
    #     data_root="./data/iphone_1x_dev/paper-windmill/",
    #     #
    #     cfg="./configs/iphone/iphone_base.yaml",
    #     device=torch.device("cuda"),
    #     tto_flag=True,
    # )

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", type=str, required=True)
    args.add_argument("--root", "-r", type=str, required=True)
    args.add_argument("--src", "-s", type=str, required=True)
    args.add_argument("--device", type=str, default="cuda:0")
    args.add_argument("--tto", action="store_true")
    args.add_argument("--skip_test_gen", action="store_true")

    args = args.parse_args()

    test_main(
        cfg=args.config,
        saved_dir=args.root,
        data_root=args.src,
        device=torch.device(args.device),
        tto_flag=args.tto,
        skip_test_gen=args.skip_test_gen,
    )
