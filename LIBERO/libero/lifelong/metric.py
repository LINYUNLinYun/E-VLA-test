"""
LIBERO lifelong 的评估指标实现，只有两件事：

1. 成功率评估：让当前策略在每个任务上跑完整条 episode，统计成功次数
   （evaluate_success / evaluate_multitask_training_success
     -> evaluate_one_task_success）
2. 损失评估：在（见过的）数据集上重算一遍 BC 损失（evaluate_loss）

另外提供 raw_obs_to_tensor_obs，把向量环境吐出来的原始观测打包成策略要的张量格式。
"""

import copy
import gc
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, cast

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.utils import safe_device

if TYPE_CHECKING:
    # 只在静态分析时导入：algos/base.py 运行时 `from libero.lifelong.metric import *`，
    # 真的 import 会形成循环导入
    from libero.libero.benchmark import Benchmark, Task
    from libero.lifelong.algos.base import Sequential
    from libero.lifelong.algos.packnet import PackNet


# 关于 cfg 一律标注成 Any 的说明：cfg 是 EasyDict（见 main.py 里的
# `cfg = EasyDict(yaml.safe_load(yaml_config))`），字段全部来自 configs/ 下的 yaml，
# Python 侧不存在对应的类定义，静态分析无法给出更精确的类型。

# 向量环境一次返回的原始观测：形状 (env_num,) 的 object ndarray，每个元素是
# {robosuite 观测名: np.ndarray} 的 dict。之所以是 ndarray 而不是 list，
# 是因为 env.step()/set_init_state() 内部做了 `np.stack(obs_list)`（见 envs/venv.py）
RawObs = np.ndarray

# 打包后喂给 policy 的输入，结构见 raw_obs_to_tensor_obs 的 docstring
TensorObs = Dict[str, Any]

# 评估时按 (episode, step) 记录的仿真状态，用于回放和可视化
SimStates = List[List[np.ndarray]]

VectorEnv = Union[DummyVectorEnv, SubprocVectorEnv]


def raw_obs_to_tensor_obs(obs: RawObs, task_emb: torch.Tensor, cfg: Any) -> TensorObs:
    """
    Prepare the tensor observations as input for the algorithm.

    把"向量环境的一次观测"翻译成"策略一次前向要的 batch"。

    Args:
        obs: (env_num,) 的 object ndarray，每个元素是单个环境的观测 dict
        task_emb: (1, D) 的任务语言指令嵌入，会被复制成 (env_num, D)
        cfg: 全局配置

    Returns:
        {"obs": {obs_name: (env_num, ...) 的张量}, "task_emb": (env_num, D)}
    """
    env_num = len(obs)

    data: TensorObs = {
        "obs": {},
        # 同一轮评估里 20 个并行环境跑的是同一个任务，语言指令嵌入完全一样，
        # 所以复制 env_num 份对齐 batch 维度
        "task_emb": task_emb.repeat(env_num, 1),                # 因为指令的嵌入是不变的所以，一次复制20份然后对齐batch
    }

    # 按模态配置（rgb / depth / low_dim）把要用的观测项都登记成空 list，后面逐个 append。
    # modality 是 {模态名: [观测项名, ...]} 的两层结构，这里拍平成一层的 all_obs_keys，
    # 因为后面逐个环境处理时并不关心某个观测项属于哪个模态
    all_obs_keys: List[str] = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    # 逐个并行环境、逐个观测项地取值：
    #   obs_key_mapping 把策略侧的逻辑名（如 agentview_rgb）翻译成 robosuite 的原生名
    #                   （如 agentview_image），这样换相机/换机器人只需要改 yaml
    #   process_obs 是 robomimic 的模态专用预处理（图像按注册好的方式缩放/转换），
    #               low-dim 走过直通；它依赖 ObsUtils 已经被同一份 obs.modality 初始化过
    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    # 每个观测项都从 [env_0, env_1, ...] 堆成 (env_num, ...) 的批量张量
    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    # data 是 {"obs": {...}, "task_emb": ...} 的嵌套 dict，普通的 .to() 处理不了，
    # map_tensor 会递归遍历到每个叶子张量再搬到 cfg.device
    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(
    cfg: Any,
    algo: "Sequential",
    task: "Task",
    task_emb: torch.Tensor,
    task_id: int,
    sim_states: Optional[SimStates] = None,
    task_str: str = "",
) -> float:
    """
    Evaluate a single task's success rate

    整体流程：建 env_num 个并行环境 -> 灌入固定的初始状态 -> 循环 max_steps 步，
    每步 {观测转张量 -> 策略出动作 -> 环境前进一步} -> 统计成功率。

    Args:
        algo: 基类 algos/base.py 的 Sequential；具体子类由字符串注册表动态实例化
              （见 evaluate.py 的 algo_map + eval()），所以这里只能按基类标注
        task: benchmark 里的任务条目（NamedTuple），提供 bddl_file 等路径信息
        task_emb: (1, D) 的任务语言指令嵌入
        sim_states: if not None, will keep track of all simulated states during
                    evaluation, mainly for visualization and debugging purpose
        task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            # get_eval_algo 只有 PackNet 有，声明成基类后解析不到，需要向下转型。
            # PackNet 是多任务共享一个网络、靠掩码隔离任务的，评估某个任务前必须先把
            # 不属于该任务的权重屏蔽掉，所以这里换成一个"掩码已冻结"的副本
            algo = cast("PackNet", algo).get_eval_algo(task_id)

        # 切到推理模式（关掉 dropout/batch norm 的训练行为），注意此时还是 torch grad 开启的
        algo.eval()
        # n_eval 是总共要评估的 episode 数，env_num 是并行环境数（不并行就是 1），
        # eval_loop_num 是把这些 episode 分批跑完需要几轮；除法是向上取整
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # 建评估用的环境：指定任务对应的 bddl 文件 + 相机分辨率
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        # 和上面完全重复的一次计算，上游遗留写法，保留原样
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # 建环境偶尔会因为渲染/共享内存的问题失败（原注释说的 frame buffer issue），
        # 失败就睡 5 秒重试，最多 5 次
        env_creation = False

        env: VectorEnv  # 只在 try 分支里赋值，这里先声明类型
        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    # 单环境用 DummyVectorEnv（同进程循环），省掉子进程开销
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    # 多环境用 SubprocVectorEnv，每个 env 一个子进程，真正并行渲染
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### Evaluation loop
        # 读入预先存好的初始状态，让每次评估都从同一批位姿开始，
        # 否则成功率会随初始状态抖动，无法比较
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in range(eval_loop_num):
            # 先把环境重置（清掉上一轮残留的物理状态），再把固定初始状态灌进去
            env.reset()
            # 这一轮 env_num 个环境分别用哪几个初始状态；取模是为了 episode 数超过
            # 初始状态数时能循环复用，同时防止下标越界
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones: List[bool] = [False] * env_num  # 每条轨迹是否已经成功（累积）
            steps = 0
            algo.reset()  # 清空策略内部的历史（有记忆的网络需要）
            obs: RawObs = env.set_init_state(init_states_)

            # 空跑 5 步全零动作，让物理引擎先稳定下来（物体落定、夹爪就位），
            # 这几步不计入步数、也不做决策
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            if task_str != "":
                # 记录仿真状态是为了事后回放/可视化（出问题时能看看到底发生了什么）
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                        sim_states[i * env_num + k].append(sim_state[k])

            while steps < cfg.eval.max_steps:
                steps += 1

                # 核心三步：观测打包 -> 策略推理 -> 环境前进
                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)

                # record the sim states for replay purpose
                if task_str != "":
                    sim_state = env.get_sim_state()
                    for k in range(env_num):
                        if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                            sim_states[i * env_num + k].append(sim_state[k])

                # 累积"已成功"标记：done 只在触发的**那一步**为 True（环境随后就自动
                # 重置了），而 done 之后环境会回到初始状态，所以要把它记住而不是每步覆盖，
                # 否则一条轨迹成功了又继续跑，最后会被判失败
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

                # 这一批全成功就不用继续跑了，提前结束
                if all(dones):
                    break

            # 统计成功数：只算前 n_eval 条，最后一批可能是没填满的部分（多出来的
            # 环境是凑数并行用的，不算分）
            for k in range(env_num):
                if i * env_num + k < cfg.eval.n_eval:
                    num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()  # 关掉子进程，否则多任务评估下来会攒一堆僵尸进程
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate


def evaluate_success(
    cfg: Any,
    algo: "Sequential",
    benchmark: "Benchmark",
    task_ids: Sequence[int],
    result_summary: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Evaluate the success rate for all task in task_ids.

    对 task_ids 里的每个任务各跑一次成功率评估，返回成功率的数组。

    Args:
        benchmark: libero.libero.benchmark 里的 Benchmark 实例，提供 get_task/
                   get_task_emb
        task_ids: 要评估的任务编号
        result_summary: 采集 sim_states 时用的汇总字典，结构由 main.py 构造
    """
    algo.eval()
    successes: List[float] = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        # 形如 "k9_p3"：k 后面是当前学到的最后一个任务，p 后面是被评估的任务。
        # 这样在持续学习里一眼能看出"学到第 k 个任务时、第 p 个任务的表现"，
        # 也是 result_summary 里存 sim_states 的 key
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(
            cfg, algo, task_i, task_emb, i, sim_states=curr_summary, task_str=task_str
        )
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(
    cfg: Any,
    algo: "Sequential",
    benchmark: "Benchmark",
    task_ids: Sequence[int],
) -> np.ndarray:
    """
    Evaluate the success rate for all task in task_ids.

    和 evaluate_success 的唯一区别：不记录 sim_states。
    多任务训练时所有任务是一起学的，没有"学到第几个任务"这个概念，
    所以也不需要按 k_p 命名去存回放数据。
    """
    algo.eval()
    successes: List[float] = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


@torch.no_grad()  # 纯评估，不需要建计算图，省显存也更快
def evaluate_loss(
    cfg: Any,
    algo: "Sequential",
    benchmark: "Benchmark",
    datasets: Sequence[Dataset],
) -> np.ndarray:
    """
    Evaluate the loss on all datasets.

    逐个数据集算一遍 BC 损失，返回每个任务的 loss 数组
    （在 main.py 里作为持续学习"遗忘程度"指标：L_fwd / 混淆矩阵的一行）。

    Args:
        datasets: 训练/评估用的序列数据集（SequenceVLDataset 等，均为 torch Dataset）
    """
    algo.eval()
    losses: List[float] = []
    for i, dataset in enumerate(datasets):
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            # 同 evaluate_one_task_success，get_eval_algo 是 PackNet 独有的，
            # 这里换成掩码冻结后的副本再算 loss
            algo = cast("PackNet", algo).get_eval_algo(task_id=i)

        # shuffle=False：评估要可复现，且这里只关心平均损失，顺序无关
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            shuffle=False,
        )
        test_loss = 0
        for data in dataloader:
            # 同 raw_obs_to_tensor_obs 的收尾：递归把 batch 里的张量搬到 device
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=cfg.device)
            )
            loss = algo.policy.compute_loss(data)
            test_loss += loss.item()
        # 除以 batch 数得到这个数据集上的平均损失
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
