"""Train and evaluate a custom lifelong-learning policy on LIBERO.

This is the script version of ``quick_guide_algo.ipynb``. It keeps the example
model and algorithm importable, while dataset loading and training happen only
inside ``main``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# These must be configured before importing the rendering / tokenizer stack.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import imageio.v2 as imageio
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import trange

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.language_modules import *  # noqa: F403
from libero.lifelong.models.modules.rgb_modules import *  # noqa: F403
from libero.lifelong.models.modules.transformer_modules import *  # noqa: F403
from libero.lifelong.models.policy_head import *  # noqa: F403
from libero.lifelong.utils import create_experiment_dir, get_task_embs, safe_device


class ExtraModalityTokens(nn.Module):
    """Map each enabled proprioceptive modality to an embedding token."""

    MODALITIES = (
        (7, "use_joint", "joint_states"),
        (2, "use_gripper", "gripper_states"),
        (3, "use_ee", "ee_states"),
    )

    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        super().__init__()
        enabled = {
            "use_joint": use_joint,
            "use_gripper": use_gripper,
            "use_ee": use_ee,
        }
        self.enabled_modalities = [
            (dimension, name)
            for dimension, flag, name in self.MODALITIES
            if enabled[flag]
        ]
        if not self.enabled_modalities:
            raise ValueError("At least one extra input modality must be enabled")

        self.encoders = nn.ModuleDict(
            {
                name: self._make_mlp(
                    dimension,
                    extra_num_layers,
                    extra_hidden_size,
                    extra_embedding_size,
                )
                for dimension, name in self.enabled_modalities
            }
        )

    @staticmethod
    def _make_mlp(input_size, num_layers, hidden_size, output_size):
        if num_layers <= 0:
            return nn.Sequential(nn.Linear(input_size, output_size))

        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(1, num_layers):
            layers.extend(
                [nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)]
            )
        layers.append(nn.Linear(hidden_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, obs_dict):
        tokens = [self.encoders[name](obs_dict[name]) for _, name in self.enabled_modalities]
        return torch.stack(tokens, dim=-2)


class MyTransformerPolicy(BasePolicy):
    """Transformer policy from the notebook, usable as a regular module."""

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        embed_size = policy_cfg.embed_size

        self.image_encoders = nn.ModuleDict()
        for name, input_shape in shape_meta["all_shapes"].items():
            if "rgb" not in name and "depth" not in name:
                continue
            kwargs = policy_cfg.image_encoder.network_kwargs
            kwargs.input_shape = input_shape
            kwargs.output_size = embed_size
            kwargs.language_dim = policy_cfg.language_encoder.network_kwargs.input_size
            encoder_class = globals()[policy_cfg.image_encoder.network]
            self.image_encoders[name] = encoder_class(**kwargs)

        policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        language_encoder_class = globals()[policy_cfg.language_encoder.network]
        self.language_encoder = language_encoder_class(
            **policy_cfg.language_encoder.network_kwargs
        )
        self.extra_encoder = ExtraModalityTokens(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        position_class = globals()[policy_cfg.temporal_position_encoding.network]
        self.temporal_position_encoding_fn = position_class(
            **policy_cfg.temporal_position_encoding.network_kwargs
        )
        self.temporal_transformer = TransformerDecoder(  # noqa: F405
            input_size=embed_size,
            num_layers=policy_cfg.transformer_num_layers,
            num_heads=policy_cfg.transformer_num_heads,
            head_output_size=policy_cfg.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
            dropout=policy_cfg.transformer_dropout,
        )

        policy_cfg.policy_head.network_kwargs.input_size = embed_size
        policy_cfg.policy_head.network_kwargs.output_size = shape_meta["ac_dim"]
        policy_head_class = globals()[policy_cfg.policy_head.network]
        self.policy_head = policy_head_class(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs,
        )
        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

    def temporal_encode(self, features):
        features = features + self.temporal_position_encoding_fn(features).unsqueeze(1)
        original_shape = features.shape
        self.temporal_transformer.compute_mask(features.shape)
        features = TensorUtils.join_dimensions(features, 1, 2)
        features = self.temporal_transformer(features).reshape(*original_shape)
        return features[:, :, 0]

    def spatial_encode(self, data):
        extra = self.extra_encoder(data["obs"])
        batch_size, sequence_length = extra.shape[:2]
        text = self.language_encoder(data).view(batch_size, 1, 1, -1)
        encoded = [text.expand(-1, sequence_length, -1, -1), extra]

        for image_name, encoder in self.image_encoders.items():
            images = data["obs"][image_name]
            batch_size, sequence_length, channels, height, width = images.shape
            language = (
                data["task_emb"]
                .reshape(batch_size, 1, -1)
                .repeat(1, sequence_length, 1)
                .reshape(batch_size * sequence_length, -1)
            )
            image_token = encoder(
                images.reshape(batch_size * sequence_length, channels, height, width),
                langs=language,
            ).view(batch_size, sequence_length, 1, -1)
            encoded.append(image_token)
        return torch.cat(encoded, dim=-2)

    def forward(self, data):
        return self.policy_head(self.temporal_encode(self.spatial_encode(data)))

    def get_action(self, data):
        self.eval()
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            self.latent_queue.append(self.spatial_encode(data))
            self.latent_queue = self.latent_queue[-self.max_seq_len :]
            features = self.temporal_encode(torch.cat(self.latent_queue, dim=1))
            distribution = self.policy_head(features[:, -1])
        action = distribution.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.latent_queue.clear()


class MyLifelongAlgo(Sequential):
    """Minimal custom sequential algorithm that records completed datasets."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.datasets = []
        self.policy = MyTransformerPolicy(cfg, cfg.shape_meta)

    def end_task(self, dataset, task_id, benchmark):
        del task_id, benchmark
        self.datasets.append(dataset)


def build_config(args):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "libero" / "configs"), version_base=None
    ):
        hydra_cfg = compose(config_name="config")
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.benchmark_name = args.benchmark
    cfg.data.task_order_index = args.task_order
    cfg.eval.num_procs = 1
    cfg.eval.n_eval = args.n_eval
    cfg.train.n_epochs = args.epochs
    return cfg


def prepare_datasets(cfg):
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    datasets, descriptions = [], []
    shape_meta = None
    for task_id in range(benchmark.n_tasks):
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=task_id == 0,
            seq_len=cfg.data.seq_len,
        )
        datasets.append(dataset)
        descriptions.append(benchmark.get_task(task_id).language)

    task_embeddings = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embeddings)
    datasets = [
        SequenceVLDataset(dataset, embedding)
        for dataset, embedding in zip(datasets, task_embeddings)
    ]
    return benchmark, datasets, shape_meta


def train(cfg, benchmark, datasets):
    create_experiment_dir(cfg)
    algorithm = safe_device(MyLifelongAlgo(benchmark.n_tasks, cfg), cfg.device)
    result = {
        "L_conf_mat": np.zeros((benchmark.n_tasks, benchmark.n_tasks)),
        "S_conf_mat": np.zeros((benchmark.n_tasks, benchmark.n_tasks)),
        "L_fwd": np.zeros(benchmark.n_tasks),
        "S_fwd": np.zeros(benchmark.n_tasks),
    }
    group_size = cfg.data.task_group_size

    for task_id in trange(benchmark.n_tasks, desc="Training tasks"):
        algorithm.train()
        success_auc, loss_auc = algorithm.learn_one_task(
            datasets[task_id], task_id, benchmark, result
        )
        result["S_fwd"][task_id] = success_auc
        result["L_fwd"][task_id] = loss_auc
        if cfg.eval.eval:
            algorithm.eval()
            result["L_conf_mat"][task_id, : task_id + 1] = evaluate_loss(
                cfg, algorithm, benchmark, datasets[: task_id + 1]
            )
            result["S_conf_mat"][task_id, : task_id + 1] = evaluate_success(
                cfg, algorithm, benchmark, list(range((task_id + 1) * group_size))
            )
        torch.save(result, Path(cfg.experiment_dir) / "result.pt")
    return algorithm, result


def load_forward_curves(experiment_dir: Path, cfg, n_tasks: int, seeds=(10000,)):
    num_points = cfg.train.n_epochs // cfg.eval.eval_every + 1
    curves = np.zeros((n_tasks, num_points, len(seeds)))
    for task_id in range(n_tasks):
        for seed_index, _seed in enumerate(seeds):
            path = experiment_dir / f"task{task_id}_auc.log"
            if not path.is_file():
                print(f"[warning] Missing forward curve: {path}")
                continue
            success = torch.load(path, map_location="cpu")["success"]
            best_index = success.argmax()
            success[best_index:] = success[best_index]
            curves[task_id, :, seed_index] = success
    return curves


def compute_metrics(confusion_matrices, forward_curves):
    """Compute forward transfer, backward transfer, and overall AUC."""
    num_tasks, num_seeds = confusion_matrices.shape[1:]
    backward_transfer, auc = [], []
    for seed in range(num_seeds):
        seed_bwt = seed_auc = 0.0
        for task_id in range(num_tasks):
            later = confusion_matrices[task_id + 1 :, task_id, seed]
            task_bwt = (
                confusion_matrices[task_id, task_id, seed] - later.mean()
                if len(later)
                else 0.0
            )
            task_auc = (
                later.sum() + forward_curves[task_id, :, seed].mean()
            ) / (num_tasks - task_id)
            seed_bwt += task_bwt
            seed_auc += task_auc
        backward_transfer.append(seed_bwt / num_tasks)
        auc.append(seed_auc / num_tasks)
    return {
        "fwt": forward_curves.mean(axis=(0, 1)),
        "bwt": np.asarray(backward_transfer),
        "auc": np.asarray(auc),
    }


def save_rollout(cfg, benchmark, algorithm, task_id: int, output: Path):
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv

    if not 0 <= task_id < benchmark.n_tasks:
        raise ValueError(f"rollout task must be in [0, {benchmark.n_tasks - 1}]")
    task = benchmark.get_task(task_id)
    task_embedding = benchmark.get_task_emb(task_id)
    if cfg.lifelong.algo == "PackNet":
        algorithm = algorithm.get_eval_algo(task_id)
    algorithm.eval()

    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
    frames = []
    try:
        initial_states = torch.load(
            os.path.join(
                cfg.init_states_folder, task.problem_folder, task.init_states_file
            )
        )
        env.reset()
        observations = env.set_init_state(initial_states[:1])
        algorithm.reset()
        for _ in range(5):
            observations, _, _, _ = env.step(np.zeros((1, 7)))

        for _ in range(cfg.eval.max_steps):
            data = raw_obs_to_tensor_obs(observations, task_embedding, cfg)
            action = algorithm.policy.get_action(data)
            observations, _, done, _ = env.step(action)
            frames.append(observations[0]["agentview_image"][::-1])
            if done[0]:
                break
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Saved rollout to {output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="libero_object",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    )
    parser.add_argument("--task-order", type=int, default=0, choices=range(22))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n-eval", type=int, default=5)
    parser.add_argument(
        "--rollout-task", type=int, help="After training, save a rollout for this task."
    )
    parser.add_argument(
        "--rollout-output", type=Path, default=REPO_ROOT / "tmp_video.mp4"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    benchmark, datasets, shape_meta = prepare_datasets(cfg)
    cfg.shape_meta = shape_meta
    cfg.policy.policy_type = "MyTransformerPolicy"
    cfg.lifelong.algo = "MyLifelongAlgo"

    print(f"Training {benchmark.n_tasks} tasks; output will be created by LIBERO.")
    algorithm, result = train(cfg, benchmark, datasets)
    print("Success confusion matrix:\n", result["S_conf_mat"])
    print("Forward success AUC:\n", result["S_fwd"])

    curves = load_forward_curves(
        Path(cfg.experiment_dir), cfg, benchmark.n_tasks
    )
    confusion = result["S_conf_mat"][..., np.newaxis]
    print("Metrics:", compute_metrics(confusion, curves))
    if args.rollout_task is not None:
        save_rollout(
            cfg, benchmark, algorithm, args.rollout_task, args.rollout_output
        )


if __name__ == "__main__":
    main()
