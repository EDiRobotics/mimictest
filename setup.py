from setuptools import setup, find_packages

setup(
    name='mimictest',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'diffusers',
        'scipy',
        'einops',
        'opencv-python-headless',
        'accelerate',
        'av',
        'cython<3',
        'gymnasium',
        'numpy<1.24',
        'free-mujoco-py==2.1.6',
        'stable-baselines3',
        'robomimic',
    ],
)

