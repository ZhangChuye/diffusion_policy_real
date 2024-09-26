import time
import hydra
import math
from glob import glob
import pandas as pd
import torch

import datetime
import torch

torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

import torch.distributed as dist
import lightning
from lightning.fabric import Fabric
import gc

import os
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import wandb
import cv2  # Import OpenCV
import click

from atm.utils.env_utils import build_single_env

from einops import rearrange
import threading
import dill


import pathlib
import skvideo.io
import scipy.spatial.transform as st


# from utils.real_inference_utils import get_real_obs_resolution, get_real_obs_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace


import pygame

HEADLESS = False

fn = [
    # "./data/atm_libero/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo/env_meta.json",
    # "/home/ye/ATM_API/data/atm_libero/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo/env_meta.json",
    "/home/ye/diffusion_policy_real/data/spatial_meta/env_meta.json"
    # "./data/atm_libero/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo/env_meta.json",
    # "./data/atm_libero/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo/env_meta.json",
    # "./data/atm_libero/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo/env_meta.json",
    # "./data/atm_libero/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo/env_meta.json",
    # "./data/atm_libero/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo/env_meta.json",
    # "./data/atm_libero/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo/env_meta.json",
]

name = [
    # "libero_object",
    "libero_spatial",
    # "libero_object",
    # "libero_object",
    # "libero_object",
    # "libero_object",
    # "libero_object",
    # "libero_object",
]

task_name = [
    # "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    # "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    # "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    # "pick_up_the_ketchup_and_place_it_in_the_basket",
    # "pick_up_the_milk_and_place_it_in_the_basket",
    # "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    # "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
]


def rearrange_videos(videos, success, success_vid_first, fail_vid_first):
    success = np.array(success)
    rearrange_idx = np.arange(len(success))
    if success_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[success_idx], videos[fail_idx]], axis=0)
        rearrange_idx = np.concatenate([success_idx, fail_idx], axis=0)
    if fail_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[fail_idx], videos[success_idx]], axis=0)
        rearrange_idx = np.concatenate([fail_idx, success_idx], axis=0)
    return videos, rearrange_idx


def render_done_to_boundary(frame, success, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (b, c, h, w)
        success: (b, 1)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    if any(success):
        b, c, h, w = frame.shape
        color = np.array(color, dtype=frame.dtype)[None, :, None, None]
        boundary = int(min(h, w) * 0.015)
        frame[success, :, :boundary, :] = color
        frame[success, :, -boundary:, :] = color
        frame[success, :, :, :boundary] = color
        frame[success, :, :, -boundary:] = color
    return frame


class DPActionInput:
    def __init__(self, checkpoint, output_dir, device):
        self.a = np.zeros(7)

        self.horizon = 60000000
        self.env = build_single_env(
            img_size=128,
            env_type="libero",
            env_meta_fn=fn,
            env_name=name,
            task_name=task_name,
            render_gpu_ids=[0],
            vec_env_num=1,
        )

        self.obs = self.env.reset()
        self.done = False
        self.episode_frames = []
        self.info = None

        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.device = device
        
        self._setup_policy()
        
        self._setup_obs_buffer()
        
        self.env_step = 0
        
    def _setup_policy(self):
        steps_per_inference = 16
        # print("enter")
        if os.path.exists(self.output_dir):
            click.confirm(f"Output path {self.output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        
        print("checkpoint: ", self.checkpoint)
        video_capture_fps = 30
        max_obs_buffer_size = 30
        
        # load checkpoint
        ckpt_path = self.checkpoint
        payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        cfg._target_ = cfg._target_
        cfg.policy._target_ = cfg.policy._target_
        cfg.ema._target_ = cfg.ema._target_

        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # policy
        self.policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        # device = torch.device("cuda:0")
        device = torch.device(self.device)
        self.policy.eval().to(device)
        self.policy.reset()

        ## set inference params
        self.policy.num_inference_steps = 16  # DDIM inference iterations
        self.policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

        # obs_res = get_real_obs_resolution(cfg.task.shape_meta)
        n_obs_steps = cfg.n_obs_steps
        print("n_obs_steps: ", n_obs_steps)
        print("steps_per_inference:", steps_per_inference)

    def _setup_obs_buffer(self):
        self.agent_view_buffer = np.zeros((1, 2, 3, 128, 128))
        self.eye_in_hand_buffer = np.zeros((1, 2, 3, 128, 128))
        self.ee_pos_buffer = np.zeros((1, 2, 3))
        self.ee_ori_buffer = np.zeros((1, 2, 3))
        self.gripper_states_buffer = np.zeros((1, 2, 2))
        self.joint_states_buffer = np.zeros((1, 2, 7))
        
    def buffer2dict(self):
        obs_dict = dict()
        obs_dict["agentview"] = torch.from_numpy(self.agent_view_buffer).to(self.device)
        obs_dict["eye_in_hand"] = torch.from_numpy(self.eye_in_hand_buffer).to(self.device)
        # obs_dict["ee_pos"] = torch.from_numpy(self.ee_pos_buffer).to(self.device)
        # obs_dict["ee_ori"] = torch.from_numpy(self.ee_ori_buffer).to(self.device)
        obs_dict["gripper_states"] = torch.from_numpy(self.gripper_states_buffer).to(self.device)
        obs_dict["joint_states"] = torch.from_numpy(self.joint_states_buffer).to(self.device)
        return obs_dict
    
    
    def new_obs(self, obs):
        # debug new_obs
        # print("^"*20)
        # print(f"step: {self.env_step}")
        # print(f"entered gripper state: {obs['robot0_gripper_qpos']}")
        # print(f"gripper state buffer: {self.gripper_states_buffer}")
        
        # move buffer
        self.agent_view_buffer = np.roll(self.agent_view_buffer, shift=-1, axis=1)
        self.eye_in_hand_buffer = np.roll(self.eye_in_hand_buffer, shift=-1, axis=1)
        self.ee_pos_buffer = np.roll(self.ee_pos_buffer, shift=-1, axis=1)
        self.ee_ori_buffer = np.roll(self.ee_ori_buffer, shift=-1, axis=1)
        self.gripper_states_buffer = np.roll(self.gripper_states_buffer, shift=-1, axis=1)
        self.joint_states_buffer = np.roll(self.joint_states_buffer, shift=-1, axis=1)
        
        # print("-"*20)
        # print(f"shifted gripper state buffer: {self.gripper_states_buffer}")
        
        # reshape 1, c_id, h, w, c -> 1, c_id, c, h, w        
        agent_img = obs["agentview_image"]
        agent_img = rearrange(agent_img, "b h w c -> b c h w")
        eye_in_hand_img = obs["robot0_eye_in_hand_image"]
        eye_in_hand_img = rearrange(eye_in_hand_img, "b h w c -> b c h w")
        # update buffer
        self.agent_view_buffer[0, 1] = agent_img[0]
        self.eye_in_hand_buffer[0, 1] = eye_in_hand_img[0]
        # reverse the last dim of obs
        # obs["robot0_eef_pos"] = 
        # self.ee_pos_buffer[0, 1] = obs["robot0_eef_pos"] #obs["robot0_eef_pos"]
        # quat2euler
        # self.ee_ori_buffer[0, 1] = st.Rotation.from_quat(obs["robot0_eef_quat"]).as_euler("xyz")
        self.gripper_states_buffer[0, 1] = obs["robot0_gripper_qpos"]
        self.joint_states_buffer[0, 1] = obs["robot0_joint_pos"]
        
        # print("-"*20)
        # print(f"updated gripper state buffer: {self.gripper_states_buffer}")
        # print("^"*20)
        
        self.env_step+=1

    def run(self):
        old_a = None
        step_i = 0

        # 1 * 7 action space: len(action) == 7
        zero_action = np.zeros(7).reshape(1, 7)
        assert len(zero_action) == 1
        self.obs, self.r, self.done, self.info = self.env.step(zero_action)
        
        self.new_obs(self.obs)
        self.new_obs(self.obs)
        # exit()
        
        # printout obs key and shape
        # print("obs keys: ", self.obs.keys())
        # for key in self.obs.keys():
        #     print(key, self.obs[key].shape)
        # # print("obs: ", self.obs)
        # exit()

        while self.horizon is None or step_i < self.horizon:
            rgb = self.obs["image"]  # (b, v, h, w, c)
            task_emb = self.obs.get("task_emb", None)

            # self.a = np.clip(self.a, -1, 1)
            # act_vec = self.a[None, ...]
            obs_dict = self.buffer2dict()
            # run inference
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)
                action = result["action"][0].detach().to("cpu").numpy()
                print(f"inference once in env step {self.env_step}")
                # print(action)

            # if a is not too small, step env
            # if np.linalg.norm(self.a) > 0.1:
            
            # for act_horizon in range(action.shape[0]):
            for act_horizon in range(8):
                this_act = action[act_horizon]
                # reverse the last dim of this act
                
                this_act = this_act.reshape(1, 7)
                self.obs, self.r, self.done, self.info = self.env.step(this_act)
                self.new_obs(self.obs)
                # sleep 0.5
                # time.sleep(0.2)

                print(f"carries out 1 action in env step {self.env_step}")
                time.sleep(0.001)
            
            # self.new_obs(self.obs)
                # old_a = self.a
                # self.a = np.zeros(7)

                success = list(self.info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape

                frame = np.concatenate(
                    [
                        video_img[:, 0],
                        np.ones((b, c, h, w), dtype=np.uint8) * 255,
                        video_img[:, 1],
                    ],
                    axis=-1,
                )  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                self.episode_frames.append(frame)

                step_i += 1
                last_info = self.info

                # Resize the frame to a larger window size
                frame_rgb = rearrange(
                    rgb, "b v h w c -> b (v h) w c"
                ).copy()  # Convert to OpenCV format, if v > 1, stacks views vertically
                frame_bgr = cv2.cvtColor(
                    frame_rgb[0], cv2.COLOR_RGB2BGR
                )  # Assuming single batch
                desired_height = 1200  # Specify the desired height
                desired_width = int(
                    frame_bgr.shape[1] * (desired_height / frame_bgr.shape[0])
                )  # Maintain aspect ratio
                large_frame = cv2.resize(frame_bgr, (desired_width, desired_height))

                if self.done:
                    large_frame = cv2.putText(
                        large_frame,
                        "Episode Done",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if not HEADLESS:
                    cv2.imshow("Real-time Video", large_frame)  # Display the frame
                    cv2.waitKey(10)  # Wait for 10 ms between frames
                if self.horizon is not None and step_i >= self.horizon:
                    break

        if not HEADLESS:
            cv2.destroyAllWindows()

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(
    checkpoint,
    output_dir,
    device,
):
    dpai = DPActionInput(checkpoint=checkpoint, output_dir=output_dir, device=device)
    dpai.run()
    

if __name__ == "__main__":
    main()

    
    