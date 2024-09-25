"""
Usage:
(robodiff)$ python eval_real_robot.py -c <ckpt_path> -o <save_dir>
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os

import time
from multiprocessing.managers import SharedMemoryManager
import click
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st

import time
import math
import copy
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
import scipy.spatial.transform as st


# from utils.real_inference_utils import get_real_obs_resolution, get_real_obs_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
# @profile
def main(
    checkpoint,
    output_dir,
    device,
):
    steps_per_inference = 16
    print("enter")
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    
    print("checkpoint: ", checkpoint)
    video_capture_fps = 30
    max_obs_buffer_size = 30
    
    # load checkpoint
    ckpt_path = checkpoint
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cfg._target_ = cfg._target_
    cfg.policy._target_ = cfg.policy._target_
    cfg.ema._target_ = cfg.ema._target_

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda:0")
    policy.eval().to(device)
    policy.reset()

    ## set inference params
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    
    '''
    shape_meta:
    obs:
      agentview:
        shape:
        - 3
        - 128
        - 128
        type: rgb
      eye_in_hand:
        shape:
        - 3
        - 128
        - 128
        type: rgb
      ee_pos:
        shape:
        - 3
        type: low_dim
      ee_ori:
        shape:
        - 3
        type: low_dim
      gripper_states:
        shape:
        - 2
        type: low_dim
      joint_states:
        shape:
        - 7
        type: low_dim
    '''
    
    # rand an obs for testing
    # arg to 1 * 4 * ...
    # 4 is horizon, 1 is batch size
    
    obs_dict = dict()
    obs_dict["agentview"] = torch.rand(1, 2, 3, 128, 128).to(device)
    obs_dict["eye_in_hand"] = torch.rand(1, 2, 3, 128, 128).to(device)
    obs_dict["ee_pos"] = torch.rand(1, 2, 3).to(device)
    obs_dict["ee_ori"] = torch.rand(1, 2, 3).to(device)
    obs_dict["gripper_states"] = torch.rand(1, 2, 2).to(device)
    obs_dict["joint_states"] = torch.rand(1, 2, 7).to(device)
    
    # run inference
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        action = result["action"][0].detach().to("cpu").numpy()
        print(action)
    
    
    # # buffer
    # shm_manager = SharedMemoryManager()
    # shm_manager.start()

    # examples = dict()
    # examples["mid"] = np.empty(shape=obs_res[::-1] + (3,), dtype=np.uint8)
    # examples["right"] = np.empty(shape=obs_res[::-1] + (3,), dtype=np.uint8)
    # examples["eef_qpos"] = np.empty(shape=(7,), dtype=np.float64)
    # examples["qpos"] = np.empty(shape=(7,), dtype=np.float64)
    # examples["timestamp"] = 0.0
    # obs_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
    #     shm_manager=shm_manager,
    #     examples=examples,
    #     get_max_k=max_obs_buffer_size,
    #     get_time_budget=0.2,
    #     put_desired_frequency=video_capture_fps,
    # )
    
    # # get observation
    # k = math.ceil(n_obs_steps * (video_capture_fps / frequency))
    # last_data = obs_ring_buffer.get_last_k(k=k, out=last_data)
    # last_timestamp = last_data["timestamp"][-1]
    # obs_align_timestamps = last_timestamp - (np.arange(n_obs_steps)[::-1] * dt)

    # obs_dict = dict()
    # this_timestamps = last_data["timestamp"]
    # this_idxs = list()
    # for t in obs_align_timestamps:
    #     is_before_idxs = np.nonzero(this_timestamps <= t)[0]
    #     this_idx = 0
    #     if len(is_before_idxs) > 0:
    #         this_idx = is_before_idxs[-1]
    #     this_idxs.append(this_idx)
    # for key in last_data.keys():
    #     obs_dict[key] = last_data[key][this_idxs]

    # obs_timestamps = obs_dict["timestamp"]
    # print("Got Observation!")

    # run inference
    
    # with torch.no_grad():
    #     obs_dict_np = get_real_obs_dict(
    #         env_obs=obs_dict, shape_meta=cfg.task.shape_meta)
    #     obs_dict = dict_apply(obs_dict_np,
    #         lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda:0")))

    #     result = policy.predict_action(obs_dict)

    #     action = result["action"][0].detach().to("cpu").numpy()
    
if __name__ == "__main__":
    main()