
cd mbrl-lib
pip install -e ".[dev]"
pip uninstall -y gymnasium
pip install stable-baselines3==2.1.0 gym==0.26.2 gymnasium[mujoco]==0.29.1
pip install mujoco==2.3.7 dm_control==1.0.14
pip install dill flatten_dict gin-config kornia numpy wandb
cd ..
pip install -e .
pip install -e myosuite