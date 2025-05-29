

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


