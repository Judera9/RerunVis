

## Install

After cloning the repo:

```bash
conda create -n rerun_test python=3.10
conda activate rerun_test
pip install -e .
conda install pinocchio -c conda-forge
```

Test install:
- should show a window with a g1 jumping
- can save a converted motion npy file

```bash
python scripts/rerun_visualize.py
```

## Usage

[**IMPORTANT**] Because current used DEMO g1 data have 29 dof, the DEMO g1 urdf has 23 dof. So you should remove the following codes in `utils/data_utils.py` for your robot:

```python
joint_pos = np.delete(
    joint_pos, (13, 14, 20, 21, 27, 28), axis=1
)  # WARNING
```

---

1. Add your robot urdf and meshes in `assets/`
2. Add your own robot config in `config/`
3. Change the loaded robot config in `scripts/rerun_visualize.py`

## Data Format

* Original data format support `.csv` and `.npy` files.
* Both files is the same format described as following, this can be aware from the function `load_motions_for_rerun` in `utils/data_utils.py`

```csv
...
m_{t-1} -> ROOT_POS(3), ROOT_ORI(4), JOINT_POS(NUM_JOINTS)
m_t -> ROOT_POS(3), ROOT_ORI(4), JOINT_POS(NUM_JOINTS)
...
```

* The converted data format is `.npy` file.
* The converted data include both joint space states and keypoints states, which are calculated using Pinocchio. You can try using different frame (*LOCAL, WORLD, LOCAL_WORLD_ALIGNED*) to calculate the keypoints states.
* details can also be found in the function `load_motions_for_rerun` in `utils/data_utils.py`
