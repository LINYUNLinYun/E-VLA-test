---
base_model:
- lerobot/pi0
license: apache-2.0
pipeline_tag: robotics
library_name: transformers
---

# INTACT Probing Suite: Pi0 Fine-tuned on BridgeV2

> 📦 **This model is part of the [INTACT Probing Suite Collection](https://huggingface.co/collections/ai4ce/intact-probing-suite-684e5601e9ed640fdd9b994b)**  
> Explore other variants:
>  - [Pi0 from scratch on BridgeV2](https://huggingface.co/juexzz/INTACT-pi0-scratch-bridge)
>  - [Pi0 finetuned with paraphrase on BridgeV2](https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge)

## INTACT-pi-finetune-bridge

This repository contains a checkpoint of the Pi0 model ([HF implementation](https://huggingface.co/lerobot/pi0) | [Paper](https://arxiv.org/abs/2410.24164v1)) finetuned on the BridgeV2 dataset for robotic manipulation tasks. 
The model is later used for testing on the [Simpler Environment](https://github.com/simpler-env/SimplerEnv) and our [INTACT](https://github.com/ai4ce/INT-ACT) Probing Suite for the generalization boundaries of VLA models.

**Paper**: [From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models](https://arxiv.org/abs/2506.09930)

## Model Details

- **Base Model**: [lerobot/pi0](https://huggingface.co/lerobot/pi0)
- **Training Dataset**: [BridgeV2](https://rail-berkeley.github.io/bridgedata/)
- **Model Type**: Vision-Language-Action (VLA) model for robotics
- **Fine-tuning Method**: See our [paper](https://arxiv.org/abs/2506.09930)
- **Training Framework**: See our [repository](https://github.com/ai4ce/INT-ACT)



## Quick Start


### Usage in INTACT

```shell
git clone --recurse-submodules https://github.com/ai4ce/INT-ACT.git
cd INT-ACT
uv sync
source .venv/bin/activate
python 
```
Or directly in python with Lerobot, see blow:

### Integration with LeRobot

First, install lerobot
```bash
pip install lerobot
```
Then

```python
import torch
from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy

# Load model
policy = Pi0Policy.from_pretrained("juexzz/INTACT-pi0-finetune-bridge")

# Inference
with torch.no_grad():
    actions = policy.select_action(batch)
```


### Training Configuration
- **Training Steps**: 15 epochs ~22695 steps.
- **Batch Size**: 1024
- **Learning Rate**: 1e-5
- **Hardware**: 4 H100/A100
- **Input Modalities**: single image (to work with SimplerEnv), 1 language instruction, 1 robot state.
- **Output**: robot actions (delta EEF) with chunk size of 4.
For more details please refer to our [paper](https://arxiv.org/abs/2506.09930) and [code](https://github.com/ai4ce/INT-ACT)


## Evaluation

**Checkpoint choice**
After training 15 epochs, we sweep the checkpoint at epoch 1, 2, 3, 4, 5, 10, 15 for performance on the original 4 Bridge tasks in the SimplerEnv, and choose the checkpoint with *best average performance* for each of the three Pi0 variants.
Therefore, you may still get a better success rate for a specific task at other checkpoints.
As a result, the best checkpoint for this pi0 finetune model is at step 7565 (epoch 5).

The comparison of their performance on Simpler are shown below. 

### Performance Comparison on SimplerEnv

**Success rate** comparison on the SimplerEnv with other pi0 variants and some other baselines experimented in our INTACT suite.
For a more detailed comparison, please refer to the [paper](https://arxiv.org/abs/2506.09930).


| Model | carrot_on_plate | eggplant_in_basket | stack_cube | spoon_on_towel |
|-------|-----------------|-------------------|------------|----------------|
| **Pi0 finetune (This Model)** | 0.361 | 0.819 | 0.264 | 0.458 |
| [Pi0 finetune rephrase](https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge) | 0.500 | 0.944 | 0.222 | 0.597 |
| [Pi0 scratch](https://huggingface.co/juexzz/INTACT-pi0-scratch-bridge) | 0.542 | 0.903 | 0.403 | 0.875 |
| Spatial VLA | 0.125 | 0.958 | 0.292 | 0.208 |
| Magma | 0.250 | 0.611 | 0.097 | 0.208 |
| Octo Small | 0.014 | 0.097 | 0.000 | 0.097 |
| Octo Base | 0.014 | 0.306 | 0.000 | 0.014 |




## Citation

If you use this model in your research, please cite:

```bibtex
@article{fang2025intention,
  title={From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models},
  author={Fang, Irving and Zhang, Juexiao and Tong, Shengbang and Feng, Chen},
  journal={arXiv preprint arXiv:2506.09930},
  year={2025}
}
```

## Related Work

- **Pi0 (official)**: [pi0 (JAX)](https://github.com/Physical-Intelligence/openpi)
- **Base Model (Pi0 HF)**: [lerobot/pi0](https://huggingface.co/lerobot/pi0)
- **Dataset**: [BridgeV2](https://bridge-v2.github.io/)
- **Framework**: [LeRobot](https://github.com/huggingface/lerobot)
- **Simpler Environment**: [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
- **Open-source Pi0 Implementation by Allen Ren**: [open-pi-zero](https://github.com/allenzren/open-pi-zero)

## License

This model is released under the Apache 2.0 license. Please see the base model's license for any additional restrictions.

## Support

For questions about this model:
- 📧 Open an issue in this repository
- 💬 Discussion tab for community questions
- 📖 Check our [paper](https://arxiv.org/abs/2506.09930) for technical details

---

*Last updated: June 2025*