import os
import numpy as np
import glob
from tqdm import tqdm
from typing import Any, Optional
import pinocchio as pin


class DataUtils:
    def __init__(self):
        print("DataUtils has no class properties for now")
        pass

    @staticmethod
    def load_motions_for_rerun(
        input_dir: str,
        original_freq: int,
        target_freq: int,
        processed: bool = False,
        clip: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict, Any, Any]:
        """Load motion data from files in the specified directory.

        This function loads motion data from .csv and .npy files, processes them by:
        1. Extracting motion data from files
        2. Resampling to the target frequency
        3. Processing positions and rotations
        4. Calculating velocities using finite differences

        Args:
            input_dir: Path to directory containing motion data files (.csv or .npy)
            original_freq: Original frequency of the motion data in Hz
            target_freq: Target frequency for the resampled motion data in Hz
            processed: Flag indicating whether the data is already processed
            clip: Optional list of two floats [start, end] to clip the data

        Returns:
            tuple containing:
                - configuration_data: Array of shape [num_frames, data_dim] containing
                  positions and rotations [root_pos(3), root_rot(4), joint_data(...)]
                - velocity_data: Array of shape [num_frames, data_dim] containing
                  velocities [root_vel(6), joint_vel(...)]
                - data_info: Dictionary with metadata about loaded files including:
                  name, type, path, freq, length, and resampled_length
                - link_pos: Array of shape [num_frames, 3] containing link positions
                - link_vel: Array of shape [num_frames, 6] containing link velocities

        Raises:
            SystemExit: If the input_dir does not exist
        """
        if not os.path.exists(input_dir):
            raise SystemExit(f"Data path {input_dir} does not exist.")

        # Initialize variables
        data_list = []
        configuration_list = []
        velocity_list = []
        data_info = {
            "name": [],
            "type": [],
            "path": [],
            "freq": [],
            "length": [],
            "resampled_length": [],
        }

        # Load data
        data_files_path_list = glob.glob(os.path.join(input_dir, "**"), recursive=True)
        data_files_path_list = [
            f for f in data_files_path_list if f.endswith((".csv", ".npy"))
        ]
        for data_file_path in data_files_path_list:
            if ".csv" in data_file_path:
                csv_data = np.genfromtxt(data_file_path, delimiter=",")
                len_buf = csv_data.shape[0]
                if clip is not None:
                    csv_data = csv_data[int(clip[0] * len_buf) : int(clip[1] * len_buf)]

                data_info["name"].append(data_file_path.split("/")[-1])
                data_info["type"].append("csv")
                data_info["path"].append(data_file_path)
                data_info["length"].append(csv_data.shape[0])
                data_info["freq"].append(original_freq)
                data_list.append(csv_data)
            if ".npy" in data_file_path and not processed:
                npy_data = np.load(data_file_path)
                len_buf = npy_data.shape[0]
                if clip is not None:
                    npy_data = npy_data[int(clip[0] * len_buf) : int(clip[1] * len_buf)]

                # Update data info
                filename = data_file_path.split("/")[-1]
                data_info["name"].append(filename)
                data_info["type"].append("npy")
                data_info["path"].append(data_file_path)
                data_info["length"].append(npy_data.shape[0])
                data_info["freq"].append(original_freq)
                
                data_list.append(npy_data)
            if ".npy" in data_file_path and processed:
                data_group = np.load(data_file_path, allow_pickle=True).item()
                len_buf = data_group["body_states"].shape[0]
                if clip is not None:
                    data_group["body_states"] = data_group["body_states"][
                        int(clip[0] * len_buf) : int(clip[1] * len_buf)
                    ]
                    data_group["dof_pos_vel"] = data_group["dof_pos_vel"][
                        int(clip[0] * len_buf) : int(clip[1] * len_buf)
                    ]

                # Load data
                root_pos = data_group["body_states"][:, 0, :3]
                root_quat = data_group["body_states"][:, 0, 3:7]
                root_vel = data_group["body_states"][:, 0, 7:13]
                joint_pos = data_group["dof_pos_vel"][:, :, 0]
                joint_vel = data_group["dof_pos_vel"][:, :, 1]
                link_pos = data_group["body_states"][:, :, :3]
                link_vel = data_group["body_states"][
                    :, :, 26:29
                ]  # local linear velocity
                data_info = data_group["data_info"]
                configuration_data = np.concatenate(
                    [root_pos, root_quat, joint_pos], axis=1
                )
                velocity_data = np.concatenate([root_vel, joint_vel], axis=1)

                # directly return loaded data
                return configuration_data, velocity_data, data_info, link_pos, link_vel

        # Downsample data
        for i in range(len(data_list)):
            data_list[i] = DataUtils.resample_motions_amass(
                data_list[i], data_info["freq"][i], target_freq
            )
            data_info["resampled_length"].append(data_list[i].shape[0])

        # Process each data file
        if not processed:
            for i in range(len(data_list)):
                # Extract positions and quaternions, joint positions
                root_pos = data_list[i][:, :3]
                root_quat = data_list[i][:, 3:7]
                joint_pos = data_list[i][:, 7:]
                joint_pos = np.delete(
                    joint_pos, (13, 14, 20, 21, 27, 28), axis=1
                )  # WARNING

                # Calculate velocities using finite differences
                dt = 1 / target_freq  # adjust if different

                # Initialize velocities array (6 values: 3 for linear, 3 for angular)
                root_vel = np.zeros((len(root_pos), 6))

                # Calculate velocities using Pinocchio's SE3 and log operations
                for i in range(len(root_pos) - 1):
                    pose1 = DataUtils.se3_from_pos_quat(root_pos[i], root_quat[i])
                    pose2 = DataUtils.se3_from_pos_quat(
                        root_pos[i + 1], root_quat[i + 1]
                    )
                    root_vel[i] = DataUtils.calc_root_velocity(pose1, pose2, dt)
                joint_vel = np.zeros_like(joint_pos)
                joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt

                configuration_list.append(
                    np.concatenate([root_pos, root_quat, joint_pos], axis=1)
                )
                velocity_list.append(np.concatenate([root_vel, joint_vel], axis=1))

        # Concatenate all data
        configuration_data = np.concatenate(configuration_list, axis=0)
        velocity_data = np.concatenate(velocity_list, axis=0)

        return configuration_data, velocity_data, data_info, None, None

    @staticmethod
    def convert_to_standard_npy_and_save(
        configuration_data: np.ndarray,
        velocity_data: np.ndarray,
        data_info: dict,
        robot_urdf_pth,
        joint_names,
        track_body_names,
        save_file_path,
    ) -> None:
        robot = pin.RobotWrapper.BuildFromURDF(
            robot_urdf_pth, os.path.dirname(robot_urdf_pth), pin.JointModelFreeFlyer()
        )

        save_dof_pos_vel = []
        save_body_states = []
        link_states = {}

        # Add progress bar to show processing progress
        for i in tqdm(
            range(len(configuration_data)), desc="Processing frames", unit="frame"
        ):
            q = configuration_data[i]
            v = velocity_data[i]

            link_states = DataUtils.compute_link_states(q, v, robot)
            dof_pos_vel_frame_np = np.array([q[7:], v[6:]]).transpose()
            save_dof_pos_vel.append(dof_pos_vel_frame_np)
            body_states_frame_np = np.zeros((len(track_body_names), 13 + 12))
            for j, track_body in enumerate(track_body_names):
                quat = link_states[track_body]["rotation"]
                body_states_frame_np[j] = np.concatenate(
                    [
                        link_states[track_body]["position"],  # [0:3]
                        [quat[3], quat[0], quat[1], quat[2]],  # [3:7] w, x, y, z
                        link_states[track_body]["r_linear_velocity"],  # [7:10]
                        link_states[track_body]["r_angular_velocity"],  # [10:13]
                        # link_states[track_body]["l_linear_velocity"],
                        # link_states[track_body]["l_angular_velocity"],
                        # link_states[track_body]["l_w_linear_velocity"],
                        # link_states[track_body]["l_w_angular_velocity"],
                        link_states[track_body]["r_position"],  # [13:16]
                        link_states[track_body]["l_position"],  # [16:19]
                        link_states[track_body]["6d_rotation"],  # [19:25]
                    ]
                )
            save_body_states.append(body_states_frame_np)

        # Save data with progress indication
        print("Creating data dictionary...")
        save_dict = {
            "dof_pos_vel": np.array(save_dof_pos_vel, dtype=np.float32),
            "body_states": np.array(save_body_states, dtype=np.float32),
            "dof_names": joint_names,
            "body_names": track_body_names,
            "data_info": data_info,
        }
        print(f"Saving npy file to {save_file_path} ...")
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        np.save(save_file_path, save_dict)
        print(f"Successfully saved {save_file_path}")

    @staticmethod
    def compute_link_states(q, v, robot):
        """Compute link positions, velocities, rotations, and angular velocities."""
        # Update kinematics
        pin.forwardKinematics(robot.model, robot.data, q, v)

        pelvis_pos = np.zeros(3)
        pelvis_rot = np.zeros(3)

        link_states = {}
        for frame_id in range(robot.model.nframes):

            # Get frame info
            if frame_id == 0:
                continue
            if frame_id == 1:
                pelvis_pos = pin.updateFramePlacement(
                    robot.model, robot.data, frame_id
                ).translation
                pelvis_rot = pin.Quaternion(
                    pin.updateFramePlacement(robot.model, robot.data, frame_id).rotation
                ).normalized()
                pelvis_rot = np.array(
                    [pelvis_rot[3], pelvis_rot[0], pelvis_rot[1], pelvis_rot[2]]
                )
            frame_name = robot.model.frames[frame_id].name

            # Calculate root frame (pelvis) link position and rotation, and 6D rotation
            frame_placement = pin.updateFramePlacement(
                robot.model, robot.data, frame_id
            )
            global_pos = np.array(frame_placement.translation)
            root_frame_pos = DataUtils.quat_rotate_inverse(
                pelvis_rot, global_pos - pelvis_pos
            )
            local_frame_pos = DataUtils.quat_rotate_inverse(
                DataUtils.yaw_quat(pelvis_rot), global_pos - pelvis_pos
            )
            quat = pin.Quaternion(frame_placement.rotation).normalized()

            # Extract the first two rows of the rotation matrix for 6D representation
            rotation_matrix = frame_placement.rotation
            rotation_6d = np.concatenate(
                [rotation_matrix[:, 0], rotation_matrix[:, 1]]
            ).flatten()

            # Get linear and angular velocity (local frame and root frame)
            frame_velocity_local = pin.getFrameVelocity(
                robot.model, robot.data, frame_id, pin.LOCAL
            )
            # frame_velocity_world = pin.getFrameVelocity(
            #     robot.model, robot.data, frame_id, pin.WORLD
            # )
            # frame_velocity_local_world_aligned = pin.getFrameVelocity(
            #     robot.model, robot.data, frame_id, pin.LOCAL_WORLD_ALIGNED
            # )

            link_states[frame_name] = {
                "position": global_pos,  # Global position
                "rotation": quat,
                "r_linear_velocity": frame_velocity_local.linear,
                "r_angular_velocity": frame_velocity_local.angular,
                # "l_w_linear_velocity": frame_velocity_local_world_aligned.linear,
                # "l_w_angular_velocity": frame_velocity_local_world_aligned.angular,
                # "linear_velocity": frame_velocity_world.linear,
                # "angular_velocity": frame_velocity_world.angular,
                "r_position": root_frame_pos,  # Position in root frame
                "l_position": local_frame_pos,  # Position in local frame (only yaw rotation)
                "6d_rotation": rotation_6d,
            }
        return link_states

    # Create SE3 transforms from position and quaternion
    @staticmethod
    def se3_from_pos_quat(pos, quat):
        return pin.SE3(pin.Quaternion(quat[0], quat[1], quat[2], quat[3]).matrix(), pos)

    # Calculate spatial velocity
    @staticmethod
    def calc_root_velocity(pose1, pose2, dt):
        """Calculate root velocity with linear velocity and angular velocity.

        Args:
            pose1: SE3 pose at time t
            pose2: SE3 pose at time t+1
            dt: time difference

        Returns:
            np.ndarray: 6D vector [linear_velocity(3), angular_velocity(3)] in world frame
        """
        # Calculate linear velocity directly in world frame
        linear_vel = (pose2.translation - pose1.translation) / dt

        # Calculate angular velocity using rotation matrix differentiation
        R1 = pose1.rotation  # Rotation matrix of the first frame
        R2 = pose2.rotation  # Rotation matrix of the second frame

        # Calculate relative rotation
        R_diff = R1.T @ R2  # World frame relative rotation

        # For small angle changes: (R_diff - I) / dt ≈ [w]× where [w]× is the skew-symmetric matrix of angular velocity
        skew_matrix = (R_diff - np.eye(3)) / dt

        # Extract angular velocity vector from the skew-symmetric matrix
        angular_vel = np.array(
            [
                (skew_matrix[1, 0] - skew_matrix[0, 1]) / 2,
                (skew_matrix[0, 2] - skew_matrix[2, 0]) / 2,
                (skew_matrix[2, 1] - skew_matrix[1, 2]) / 2,
            ]
        )

        # Combine linear and angular velocities in world frame
        return np.concatenate([linear_vel, angular_vel])

    @staticmethod
    def slerp(val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    @staticmethod
    def resample_motions_amass(
        motions: np.ndarray, original_freq: int, target_freq: int
    ) -> np.ndarray:
        """Resample motion data from one frequency to another using interpolation.

        Args:
            motions: Motion data with shape [num_frames, data_dim]
                    Expected format: [root_pos(3), root_rot(4), joint_data(...)]
            original_freq: Original frequency of the motion data in Hz
            target_freq: Target frequency for the resampled motion data in Hz

        Returns:
            Resampled motion data with shape [target_length, data_dim]
        """
        # Calculate target length based on frequencies
        orig_length = len(motions) - 1
        orig_duration = orig_length / original_freq  # Duration in seconds
        target_length = int(orig_duration * target_freq)  # New number of frames

        # Handle edge cases
        if target_length == orig_length:
            return motions
        if orig_length == 0:
            return np.zeros((target_length, motions.shape[1]), dtype=motions.dtype)

        # Create target frame indices (normalized from 0 to orig_length-1)
        target_times = np.linspace(0, orig_length - 1, target_length)

        # Calculate indices for interpolation
        idx_low = np.floor(target_times).astype(np.int32)
        idx_high = np.ceil(target_times).astype(np.int32)

        # Ensure indices are within valid range
        idx_high = np.minimum(idx_high, orig_length - 1)

        # Calculate blend factors
        blend_factors = target_times - idx_low

        # Initialize resampled motion array
        resampled = np.zeros((target_length, motions.shape[1]), dtype=motions.dtype)

        # Process each target frame
        for i in range(target_length):
            low_idx, high_idx = idx_low[i], idx_high[i]
            blend = blend_factors[i]

            # If same frame, no interpolation needed
            if low_idx == high_idx:
                resampled[i] = motions[low_idx]
                continue

            # Get source frames
            frame_low = motions[low_idx]
            frame_high = motions[high_idx]

            # Root position (indices 0:3) - linear interpolation
            resampled[i, :3] = DataUtils.slerp(frame_low[:3], frame_high[:3], blend)

            # Root rotation (indices 3:7) - quaternion slerp
            # Using Pinocchio's quaternion slerp
            q1 = pin.Quaternion(
                frame_low[3:7]
            )  # Pinocchio quaternion takes [x,y,z,w] format
            q2 = pin.Quaternion(frame_high[3:7])
            resampled[i, 3:7] = q1.slerp(blend, q2).normalized().coeffs()

            # Joint data and other features (indices 7+) - linear interpolation
            if motions.shape[1] > 7:
                resampled[i, 7:] = DataUtils.slerp(frame_low[7:], frame_high[7:], blend)

        return resampled

    @staticmethod
    def resample_motions_slerp(
        motions: np.ndarray, original_freq: int, target_freq: int
    ) -> np.ndarray:
        """Resample motion data from one frequency to another using interpolation.

        Args:
            motions: Motion data with shape [num_frames, data_dim]
                    Expected format: NO QUAT ROTATION DATA
            original_freq: Original frequency of the motion data in Hz
            target_freq: Target frequency for the resampled motion data in Hz

        Returns:
            Resampled motion data with shape [target_length, data_dim]
        """
        # Calculate target length based on frequencies
        orig_length = len(motions)
        orig_duration = orig_length / original_freq  # Duration in seconds
        target_length = int(orig_duration * target_freq)  # New number of frames

        # Handle edge cases
        if target_length == orig_length:
            return motions
        if orig_length == 0:
            return np.zeros((target_length, motions.shape[1]), dtype=motions.dtype)

        # Create target frame indices (normalized from 0 to orig_length-1)
        target_times = np.linspace(0, orig_length - 1, target_length)

        # Calculate indices for interpolation
        idx_low = np.floor(target_times).astype(np.int32)
        idx_high = np.ceil(target_times).astype(np.int32)

        # Ensure indices are within valid range
        idx_high = np.minimum(idx_high, orig_length - 1)

        # Calculate blend factors
        blend_factors = target_times - idx_low

        # Initialize resampled motion array
        resampled = np.zeros((target_length, motions.shape[1]), dtype=motions.dtype)

        # Process each target frame
        for i in range(target_length):
            low_idx, high_idx = idx_low[i], idx_high[i]
            blend = blend_factors[i]

            # If same frame, no interpolation needed
            if low_idx == high_idx:
                resampled[i] = motions[low_idx]
                continue

            # Get source frames
            frame_low = motions[low_idx]
            frame_high = motions[high_idx]

            resampled[i, :] = DataUtils.slerp(frame_low, frame_high, blend)

        return resampled

    @staticmethod
    def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

        Args:
            q: The quaternion in (w, x, y, z). Shape is (..., 4).
            v: The vector in (x, y, z). Shape is (..., 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (..., 3).
        """
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v, axis=-1) * q_w * 2.0
        # for two-dimensional tensors, bmm is faster than einsum
        if q_vec.shape[0] == 2:
            c = q_vec * np.dot(q_vec, v, axis=-1).reshape(-1, 1) * 2.0
        else:
            c = q_vec * np.einsum("...i,...i->...", q_vec, v) * 2.0
        return a - b + c

    @staticmethod
    def yaw_quat(quat: np.ndarray) -> np.ndarray:
        """Extract the yaw component of a quaternion.

        Args:
            quat: The orientation in (w, x, y, z). Shape is (..., 4)

        Returns:
            A quaternion with only yaw component.
        """
        shape = quat.shape
        quat_yaw = quat.copy().reshape(-1, 4)
        qw = quat_yaw[:, 0]
        qx = quat_yaw[:, 1]
        qy = quat_yaw[:, 2]
        qz = quat_yaw[:, 3]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        quat_yaw[:] = 0.0
        quat_yaw[:, 3] = np.sin(yaw / 2)
        quat_yaw[:, 0] = np.cos(yaw / 2)
        quat_yaw = DataUtils.normalize(quat_yaw)
        return quat_yaw.reshape(shape)

    @staticmethod
    def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """Normalizes a given input tensor to unit length.

        Args:
            x: Input tensor of shape (N, dims).
            eps: A small value to avoid division by zero. Defaults to 1e-9.

        Returns:
            Normalized tensor of shape (N, dims).
        """
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, None)
