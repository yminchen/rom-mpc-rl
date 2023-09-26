from setuptools import setup

setup(
    name="rom-mpc-rl",
    version="0.0.1",
    install_requires=[
#        "drake",
        "gin-config",
        "gym==0.24.1",
#        "mujoco",
#        "casadi",
    ],  # Add any other dependencies here
    python_requires=">=3",
    author="Yu-Ming Chen",
    url="https://github.com/yminchen/rom-mpc-rl.git",
    license="MIT",
)
