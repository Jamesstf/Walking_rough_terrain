# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_SRC_DIR, LEGGED_GYM_ENVS_DIR
import os, shutil

def log_files(log_dir, curr_task_path):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for task_path in [curr_task_path]:
        task_full_path = os.path.join(LEGGED_GYM_ENVS_DIR, task_path)
        for f in os.listdir(task_full_path):
            file_path = os.path.join(task_full_path, f)
            if os.path.isfile(file_path):
                shutil.copy2(file_path, os.path.join(log_dir, f))

    shutil.copy2(os.path.join(LEGGED_GYM_SRC_DIR, 'utils', 'terrain.py'), os.path.join(log_dir, 'terrain.py'))
    shutil.copy2(os.path.join(LEGGED_GYM_SRC_DIR, 'utils', 'human.py'), os.path.join(log_dir, 'human.py'))
    shutil.copy2(os.path.join(LEGGED_GYM_SRC_DIR, 'scripts', 'train.py'), os.path.join(log_dir, 'train.py'))
    shutil.copy2(os.path.join(LEGGED_GYM_SRC_DIR, 'scripts', 'play.py'), os.path.join(log_dir, 'play.py'))
    
    rsl_rl_src_dir = os.path.join(LEGGED_GYM_ROOT_DIR, '..', 'rsl_rl', 'rsl_rl')
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'algorithms', 'ppo.py'), os.path.join(log_dir, 'ppo.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'modules', 'actor_critic.py'), os.path.join(log_dir, 'actor_critic.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'modules', 'actor_critic_hierarchical.py'), os.path.join(log_dir, 'actor_critic_hierarchical.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'runners', 'on_policy_runner.py'), os.path.join(log_dir, 'on_policy_runner.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'storage', 'rollout_storage.py'), os.path.join(log_dir, 'rollout_storage.py'))

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.metrics_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process_lst = []

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)
            
    def log_metrics(self, dict):
        for key, value in dict.items():
            self.metrics_log[key].append(value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        plot_process = Process(target=self._plot)
        plot_process.start()
        self.plot_process_lst.append(plot_process)

    def _plot(self, show=True):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(20, 15))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, np.array(log["dof_pos"])[:, 1], label='measured')
        if log["dof_pos_target"]: a.plot(time, np.array(log["dof_pos_target"])[:, 1], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, np.array(log["dof_vel"])[:, 1], label='measured')
        if log["dof_vel_target"]: a.plot(time, np.array(log["dof_vel_target"])[:, 1], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(np.array(log["dof_vel"])[:, 1], np.array(log["dof_torque"])[:, 1], 'x', label='measured')
        a.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"]!=[]: a.plot(time, np.array(log["dof_torque"])[:, 1], label='measured')
        a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        a.legend()

        # draw error text
        self.calculate_metrics()
        
        textstr = ""
        for key, value in self.metrics_log.items():
            mean = np.mean(np.abs(np.array(value)))
            if "_diff" in key:
                std_dev = np.std(np.abs(np.array(value)))
                textstr += f'{key}={mean:.3f} ± {std_dev:.3f}\n'
            else:
                textstr += f'{key}={mean:.3f}\n'
            
        axs[-1, -1].text(0.05, 0.05, textstr, transform=axs[-1, -1].transAxes, fontsize=14,
                        verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
        
        fig = plt.gcf()
        if show:
            plt.show()
        return fig

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def calculate_metrics(self):
        vel_x_err = np.mean(np.abs(np.array(self.state_log['base_vel_x']) - np.array(self.state_log['command_x']))).item()
        vel_y_err = np.mean(np.abs(np.array(self.state_log['base_vel_y']) - np.array(self.state_log['command_y']))).item()
        vel_yaw_err = np.mean(np.abs(np.array(self.state_log['base_vel_yaw']) - np.array(self.state_log['command_yaw']))).item()
        # dof_pos_err = np.mean(np.abs(np.array(self.state_log['dof_pos']) - np.array(self.state_log['dof_pos_target']))).item()
        root_height = np.mean(np.array(self.state_log['base_height'])).item()
        dof_vel = np.mean(np.mean(np.abs(np.array(self.state_log['dof_vel'])))).item()
        dof_torque = np.mean(np.mean(np.abs(np.array(self.state_log['dof_torque'])))).item()
        dof_energy = dof_vel * dof_torque
        base_roll = np.mean(np.abs(np.array(self.state_log['base_roll']))).item()
        base_pitch = np.mean(np.abs(np.array(self.state_log['base_pitch']))).item()
        base_tilt = (base_roll + base_pitch) / 2
        
        metrics = {
            'vel_x_err': vel_x_err,
            'vel_y_err': vel_y_err,
            'vel_yaw_err': vel_yaw_err,
            # 'dof_pos_err': dof_pos_err,
            'root_height': root_height,
            'dof_vel': dof_vel,
            'dof_torque': dof_torque,
            'dof_energy': dof_energy,
            'base_roll': base_roll,
            'base_pitch': base_pitch,
            'base_tilt': base_tilt,
        }
        self.log_metrics(metrics)
    
    def __del__(self):
        if len(self.plot_process_lst) > 0:
            for process in self.plot_process_lst:
                process.kill()