# JetBot RL Project (Isaac Lab)

This project defines and trains JetBot-based navigation tasks using the Isaac Sim + Isaac Lab framework with reinforcement learning.
The final report can be seen [here](https://github.com/federock02/EPFL-SafeRLMazeEscaping/blob/main/SafeReinforcementLearning.pdf)


[video_simulation.webm](https://github.com/user-attachments/assets/f4d664d0-762e-4b4f-b94f-a61b453bde06)

https://github.com/user-attachments/assets/c1ab1d11-bdcf-419a-8325-966c9e84c1c9

## Project Structure

The codebase is organized into the following key directories:

* **JetBot Environments**
  All JetBot RL environments are defined in:
  `/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/jetbot/`

* **Training Configuration**
  RL agent configurations (e.g., PPO, SAC) are located in:
  `/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/jetbot/agents/`

* **Robot Definition**
  The JetBot robot model and asset description are defined in:
  `/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/jetbot.py`

* **Training & Inference Scripts**
  Scripts for training and running inference are located in:
  `/IsaacLab/scripts/reinforcement_learning/skrl/`

* **Environment Registration**
  The environment IDs required for training/inference are registered in:
  `/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/jetbot/__init__.py`

## Running Training

From the root of the `IsaacLab` directory (with your Isaac Sim environment activated), launch training with:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_prob_dual.py \
    --task Isaac-JetBot-Maze-Prob-RL-Direct-v0 \
    --num_envs 4096 \
    --headless
```

* `--task`: environment ID, e.g. `Isaac-JetBot-Maze-Prob-RL-Direct-v0`
* `--num_envs`: number of parallel environments, e.g. `4096`
* `--headless`: run without GUI

## Running Inference

Run inference using a pretrained checkpoint:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-JetBot-Maze-RL-Direct-v2 \
    --num_envs 16 \
    --checkpoint /IsaacLab/logs/skrl/jetbot_direct_maze_ppo/.../saves/your_model.pt
```

* `--checkpoint`: path to the saved model `.pt` file

---
