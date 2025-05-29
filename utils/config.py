import yaml
import os

class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # ------ Dataset Config ------ #

        self.data_path = os.path.join(os.path.dirname(__file__), "..", config["data_path"])
        self.original_freq = config["original_freq"]
        self.target_freq = config["target_freq"]
        self.is_processed = config["is_processed"]
        self.clip = config["clip"]

    # ------ URDF Config ------ #

        self.robot_name = config["robot_name"]
        self.default_pose = config["default_pose"]
        self.assets_path = os.path.join(os.path.dirname(__file__), "..", config["assets_path"])
        self.urdf_path = os.path.join(self.assets_path, config["urdf_path"])
        self.root_link_name = config["root_link_name"]
        self.joint_names = config["joint_names"]
        self.track_body_names = config["track_body_names"]

    # ------ Rerun Config ------ #

        self.visualize_rerun = config["visualize_rerun"]
        self.visualize_marker = config["visualize_marker"]

    # ------ DataConvention Config ------ #

        self.save_npy_data = config["save_npy_data"]

    