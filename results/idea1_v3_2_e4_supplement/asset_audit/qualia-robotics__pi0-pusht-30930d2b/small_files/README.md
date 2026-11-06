---
datasets: lerobot/pusht
library_name: lerobot
license: apache-2.0
model_name: pi0
pipeline_tag: robotics
tags:
- pi0
- lerobot
- robotics
---

# Model Card for pi0

<!-- Provide a quick summary of what the model is/does. -->


**π₀ (Pi0)**

π₀ is a Vision-Language-Action model for general robot control, from Physical Intelligence. The LeRobot implementation is adapted from their open source OpenPI repository.

**Model Overview**

π₀ represents a breakthrough in robotics as the first general-purpose robot foundation model developed by Physical Intelligence. Unlike traditional robots that are narrow specialists programmed for repetitive motions, π₀ is designed to be a generalist policy that can understand visual inputs, interpret natural language instructions, and control a variety of different robots across diverse tasks.

For more details, see the [Physical Intelligence π₀ blog post](https://www.physicalintelligence.company/blog/pi0).


This policy has been trained and pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot).
See the full documentation at [LeRobot Docs](https://huggingface.co/docs/lerobot/index).

---

## How to Get Started with the Model

For a complete walkthrough, see the [training guide](https://huggingface.co/docs/lerobot/il_robots#train-a-policy).
Below is the short version on how to train and run inference/eval:

### Train from scratch

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<dataset> \
  --policy.type=act \
  --output_dir=outputs/train/<desired_policy_repo_id> \
  --job_name=lerobot_training \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/<desired_policy_repo_id>
  --wandb.enable=true
```

_Writes checkpoints to `outputs/train/<desired_policy_repo_id>/checkpoints/`._

### Evaluate the policy/run inference

```bash
lerobot-record \
  --robot.type=so100_follower \
  --dataset.repo_id=<hf_user>/eval_<dataset> \
  --policy.path=<hf_user>/<desired_policy_repo_id> \
  --episodes=10
```

Prefix the dataset repo with **eval\_** and supply `--policy.path` pointing to a local or hub checkpoint.

---

## Model Details

- **License:** apache-2.0