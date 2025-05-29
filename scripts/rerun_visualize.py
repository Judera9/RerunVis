import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
import os

from utils.data_utils import DataUtils
from utils.config import Config

# Load config of corresponding robot type
config = Config("config/g1.yaml")

class RerunURDF:
    def __init__(self):
        self.name = config.robot_name
        self.robot = pin.RobotWrapper.BuildFromURDF(config.urdf_path, config.assets_path, pin.JointModelFreeFlyer())
        self.Tpose = np.array(config.default_pose).astype(np.float32)
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()

    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh

    def load_visual_mesh(self):
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]

            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(
                f"urdf_{self.name}/{parent_joint_name}",
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.01,
                ),
            )

            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(
                f"urdf_{self.name}/{parent_joint_name}/{frame_name}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=mesh.visual.vertex_colors,
                    albedo_texture=None,
                    vertex_texcoords=None,
                ),
                static=True,
            )

    def update(self, configuration=None, velocity=None, link_pos=None, link_vel=None):
        """Update robot state and visualize."""
        q = self.Tpose if configuration is None else configuration
        v = np.zeros(self.robot.model.nv) if velocity is None else velocity

        # Forward kinematics for visual update
        self.robot.framesForwardKinematics(q)

        # Compute velocities and link states
        link_states = DataUtils.compute_link_states(q, v, self.robot)

        # Visualize joint transforms and meshes
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]

            # Log joint transform
            rr.log(
                f"urdf_{self.name}/{parent_joint_name}",
                rr.Transform3D(
                    translation=joint_tf.translation,
                    mat3x3=joint_tf.rotation,
                    axis_length=0.01,
                ),
            )
            # Visualize markers
            if config.visualize_marker and frame_name in config.track_body_names:
                rr.log(
                    "world_origin",  # World origin
                    rr.Transform3D(
                        translation=[0, 0, 0],
                        mat3x3=np.eye(3),
                        axis_length=0.5,
                    ),
                )
                if link_pos is None:
                    link_pos_l = link_states[frame_name]["l_position"]
                    link_vel_l = link_states[frame_name]["r_linear_velocity"]
                    root_angular_vel = link_states[config.root_link_name]["r_angular_velocity"]
                else:
                    link_pos_l = link_pos[config.track_body_names.index(frame_name)]
                    link_vel_l = link_vel[config.track_body_names.index(frame_name)]
                    root_angular_vel = link_states[config.root_link_name]["r_angular_velocity"]
                self.visualize_marker(
                    frame_name, link_pos_l, link_vel_l, root_angular_vel
                )
                link_pos_test = link_states[frame_name]["r_position"]
                rr.log(
                    f"urdf_{self.name}/{frame_name}/r_position",
                    rr.Points3D(
                        positions=[link_pos_test],
                        colors=[(255, 255, 0, 255)],
                        radii=[0.04],
                    ),
                )  # Yellow color  # Adjust marker size as needed

    def visualize_marker(self, frame_name, link_pos, link_vel, root_angular_vel):
        # Add position marker for the link
        rr.log(
            f"urdf_{self.name}/{frame_name}/position",
            rr.Points3D(
                positions=[link_pos],
                colors=[(255, 0, 0, 255)],
                radii=[0.04],
            ),
        )  # Red color  # Adjust marker size as needed

        # For linear velocity, use Arrows3D starting from the link position
        scale_factor = 0.05  # Adjust to control overall arrow lengths
        scaled_velocity = link_vel * scale_factor
        velocity_color = [(0, 255, 0, 255)]  # Green arrows
        velocity_radii = 0.005
        if frame_name == config.root_link_name:
            scaled_velocity *= 5
            velocity_color = [(0, 0, 255, 255)]  # Blue color
            velocity_radii = 0.01

            # Add angular velocity visualization for root link
            angular_vel = root_angular_vel
            ang_scale_factor = 0.5  # Adjust scale for angular velocity
            scaled_ang_velocity = angular_vel * ang_scale_factor

            # Use a single arrow to represent the angular velocity
            if (
                np.linalg.norm(angular_vel) > 0.001
            ):  # Only show non-negligible rotations
                rr.log(
                    f"urdf_{self.name}/{frame_name}/angular_velocity",
                    rr.Arrows3D(
                        vectors=[scaled_ang_velocity * np.array([0, 0, 1])],
                        origins=[
                            link_pos + np.array([0, 0, 0.6])
                        ],  # Position it above the root link
                        colors=[(255, 0, 255, 255)],  # Magenta for angular velocity
                        radii=[0.015],
                    ),
                )

        rr.log(
            f"urdf_{self.name}/{frame_name}/linear_velocity",
            rr.Arrows3D(
                vectors=[scaled_velocity],  # The velocity vector
                origins=[link_pos],  # Starting from the link position
                colors=velocity_color,
                radii=velocity_radii,  # Adjust thickness as needed
            ),
        )


if __name__ == "__main__":

    # Handle path errors
    if not os.path.exists(config.data_path):
        print(f"Data path {config.data_path} does not exist.")
        exit(1)
    if not os.path.exists(config.assets_path):
        print(f"Assets path {config.assets_path} does not exist.")
        exit(1)
    if not os.path.exists(config.urdf_path):
        print(f"URDF path {config.urdf_path} does not exist.")
        exit(1)

    file_name = config.data_path.split("/")[-1].replace(".npy", "") + "_" + str(config.target_freq) + "fps"
    robot_type = config.robot_name

    # Load motion data
    dt = 1 / config.target_freq
    if config.is_processed:
        configuration_data, velocity_data, data_info, link_pos, link_vel = (
            DataUtils.load_motions_for_rerun(
                config.data_path, config.original_freq, config.target_freq, config.is_processed, clip=config.clip
            )
        )
    else:
        configuration_data, velocity_data, data_info, _, _ = (
            DataUtils.load_motions_for_rerun(
                config.data_path, config.original_freq, config.target_freq, config.is_processed, clip=config.clip
            )
        )

    if config.visualize_rerun:
        rr.init("Reviz", spawn=True)
        rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        rerun_urdf = RerunURDF()
        for frame_nr in range(configuration_data.shape[0]):
            print(f" - Total time: {configuration_data.shape[0] * dt:.3f}s", end=" ")
            print(f"Visualizing frame {frame_nr + 1}/{configuration_data.shape[0]}")

            rr.set_time_seconds("frame_nr", frame_nr * dt)
            configuration = configuration_data[frame_nr]
            velocity = velocity_data[frame_nr]
            if config.is_processed:
                link_pos_f = link_pos[frame_nr]
                link_vel_f = link_vel[frame_nr]
                rerun_urdf.update(configuration, velocity, link_pos_f, link_vel_f)
            else:
                rerun_urdf.update(configuration, velocity)

    if config.save_npy_data:
        DataUtils.convert_to_standard_npy_and_save(
            configuration_data,
            velocity_data,
            data_info,
            robot_urdf_pth=config.urdf_path,
            joint_names=config.joint_names,
            track_body_names=config.track_body_names,
            save_file_path=os.path.join(os.path.dirname(__file__), "../output/", file_name + '.npy'),
        )
