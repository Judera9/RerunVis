from setuptools import setup, find_packages

setup(
    name="rerun_vis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.0",
        "rerun-sdk==0.22.0",
        "trimesh",
        "matplotlib",
        "tqdm",
        "pyyaml",
    ],
    python_requires=">=3.10",
    author="Jude",
    author_email="270964416@qq.com",
    description="Rerun visualization for motion data",
    keywords="motion capture, visualization",
)
