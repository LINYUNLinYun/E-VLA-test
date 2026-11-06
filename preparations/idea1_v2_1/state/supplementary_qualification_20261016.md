# v2.1 补充模型资格核查

核查日期：2026-10-16。主实验 Xiaomi / DB-CogACT 独立推进，本文记录方案中的补充候选条件。

## GR00T-N1.7-LIBERO

固定 checkpoint revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21` 的四 suite 配置已下载到 `../metadata/isaac-gr00t/`，与先前 Spatial 配置共同保存。应按实际有效 `select_layer` 解释层数，不把动作 DiT 深度计作文本主干深度。

2026-10-16 使用项目现有 Hugging Face 客户端及正常配置访问 `nvidia/Cosmos-Reason2-2B/config.json`，返回 HTTP 401 / `GatedRepoError`，提示需要获准并认证。官方 `gr00t/model/modules/qwen3_backbone.py` 的加载路径明确需要这个 backbone。该补充候选目前因访问权限未满足而不能执行正式 Full/skip 闭环；不自动接受许可、不寻找替代凭据、不以其他 backbone 替换。没有下载 GR00T 正式模型大权重或把它计入已完成的深层主模型。

只有后续正常配置获得模型访问权限并通过 Full 复现后，才能触发方案预登记的 1000 条补充评测；当前保留缺口，主实验与 E4 不因此停止。

## 原版 CogACT

本地官方仓库 `policies/cogact` 的 README 提供 SimplerEnv 评测入口，前期已查到 Small/Base/Large 通用权重；尚无本工作区可核验并直接复现的原版 LIBERO 专用 checkpoint/协议。本轮不将 DB-CogACT 冒充原版，不对通用 OXE 权重做无依据的 LIBERO 基线，也不自动启动微调。若取得可复现权重，再按方案的额外 2612 条条件矩阵执行；目前只完成候选资格调查。
