# XCX experiments

沿用 Megatron 的 CLI、模型构建、数据管线和训练循环。自定义实现放在本目录，
优先通过现有的 ModuleSpec、模型 builder hooks 和训练 callbacks 接入。

## 入口

从仓库根目录使用已经配置好 Megatron 依赖的 Python 环境运行：

```bash
python -m experiments.xcx.training.pretrain_gpt <Megatron 参数>
```

多卡启动沿用 `python -m torch.distributed.run --module`。下面的脚本提供单卡、
mock data、4 步的 GPT baseline，不需要下载 tokenizer 或数据：

```bash
bash experiments/xcx/configs/baseline.sh
```

脚本支持 `PYTHON`、`NPROC_PER_NODE`；追加参数直接传给 Megatron，重复的普通参数
以后面的值为准。它使用 local backend 和 FP32，便于检查入口，不用于性能比较。
实际训练环境按照仓库 `skills/mcore-build-and-dependency/SKILL.md` 准备。

Torch 2.6 的普通 BF16 MoE/EARH 不需要 E8M0 dtype。Core 的
`inference/quantization/mxfp8_tensor.py` 增加局部能力检查，避免未启用 MXFP8 时因
缺少 `torch.float8_e8m0fnu` 而导入失败；真正使用 Triton MXFP8 时才检查该能力并报错。
不用增加实验开关，也不改训练计算路径。

检查保存和恢复时使用新的输出目录。第二次运行把总训练步数从 4 增加到 6，
因此显式使用当前参数重设学习率调度器；沿用原训练计划续训时无需该选项：

```bash
checkpoint_dir=$(mktemp -d /tmp/xcx-baseline.XXXXXX)
bash experiments/xcx/configs/baseline.sh \
    --save "$checkpoint_dir" --save-interval 2 --ckpt-format torch_dist
bash experiments/xcx/configs/baseline.sh \
    --save "$checkpoint_dir" --load "$checkpoint_dir" \
    --save-interval 2 --ckpt-format torch_dist --train-iters 6 \
    --override-opt_param-scheduler
```

## 异步日志

在实验入口追加 `--async-log-writer` 即可开启，默认关闭。TensorBoard 和 WandB
继续使用原来的日志参数，例如：

```bash
bash experiments/xcx/configs/baseline.sh \
    --async-log-writer --tensorboard-dir /tmp/xcx-tensorboard \
    --tensorboard-log-interval 1
```

- `logging/async_writer.py` 提供 `AsyncSummaryWriter` 和 `AsyncWandbWriter`。
  TensorBoard 的 `add_scalar/add_text`、WandB 的 `log` 按调用顺序在后台执行。
  其他方法等待之前的写入完成后同步执行，保持原有调用接口。
- 入队前复制数据，tensor 会 detach 并同步复制到 CPU，因此后台不会持有计算图
  或读取后续修改的数据。异步部分是 writer 调用，GPU 指标计算和 CPU 拷贝仍在前台。
- 每个 writer 使用一个线程，最多排队 1024 次调用；队列满时前台等待。
  `flush/close/finish` 等待已提交的写入，后台异常会在后续调用或退出时抛出。
- `logging/writers.py` 在实验入口中安装 adapter，正常返回、Python 异常和
  `SystemExit` 时都会关闭 adapter 并还原 writer。官方已经调用过的 WandB
  `finish` 不会重复执行。
- 异步 logger 在 `megatron/training/global_vars.py` 增加两个通用 setter；
  没有从主体代码反向导入实验代码。其余通用启动适配见下方 KL Aux 说明。

## 基础指标与 logger

仍使用旧的 `--log-hidden-states`、`--log-params`、`--log-moe-metrics` 和
`--log-metrics-interval` 参数。默认全部关闭；开启后由官方 `Callback` 驱动，
输出到现有 TensorBoard/WandB writer，支持与 async writer 组合使用：

```bash
bash experiments/xcx/configs/baseline.sh \
    --async-log-writer --tensorboard-dir /tmp/xcx-basic-metrics \
    --log-metrics-interval 2 \
    --log-hidden-states embeddings input_layernorm attention::linear_qkv mlp \
    --log-params embedding attention::linear_qkv mlp::linear_fc1
```

| 参数 | 本批内容 |
| --- | --- |
| `--log-hidden-states` | embedding、layernorm、attention 投影/输出、MLP、shared experts 和 router logits 的统计 |
| `--log-params` | embedding/lm_head、layernorm、attention/MLP weight、router weight/bias、expert weight 的统计 |
| `--log-moe-metrics load_balance` | 完整训练步的硬负载/概率质量 MaxVio、MinVio、CV、dead expert 和低负载比例等 |
| `--log-moe-metrics router_logits_entropy ...` | logits/选中 logits/组合概率的熵及归一化熵、top-1 概率 margin 和 top-k logits margin |
| `--log-metrics-interval N` | 每 N 步采集；省略时沿用 `--tensorboard-log-interval` |

MoE 例子：在 baseline 参数后加上 `--disable-bias-linear --num-experts 4
--moe-router-topk 2 --moe-ffn-hidden-size 128`，再选择
`--log-moe-metrics load_balance router_logits_entropy router_probs_topk_entropy`。
完整 selector 列表直接定义在 `logging/logger.py`，不增加配置格式或注册层。

- `logging/logger.py`：一个 callback，负责 interval、hooks、官方 observation
  scope、全局层号和写入。`logging/metrics.py`：统计公式与紧凑记录合并。
  本批没有增加主体代码改动，Core 和训练循环保持原样。
- 保留 `hidden_states_*`、`param_*`、`load_balance/*` 等常用 tag。
  激活 std/RMS/L2 是逐 token 的特征维统计再取平均，std 使用样本标准差；
  TP 分片先合并再计算。参数按实际 weight tensor 统计，一步只采一次，排除 DP/TP 副本。
- 跨 microbatch 的激活和熵按实际观测行数加权；不等长 microbatch 时，这比旧实现
  对 microbatch 等权平均的口径更准确。完整步负载先合并原始计数再计算不均衡。
- 只有到期的训练步进行统计和通信，不采集 evaluation；复用官方重计算屏蔽，
  避免 backward 重算重复计数。跨 rank 只向日志 rank 收集紧凑记录。
- 当前覆盖 GPT decoder 及其 embedding/lm_head。CUDA graphs、GTP 权重重物化和
  FSDP 参数统计尚未接入，会明确报错；融合 layernorm 的归一化后激活未暴露时也会报错。
  具体参数和 expert 分组以当前模型实际存储的 weight tensor 为准。
- 梯度/更新诊断、逐 expert 的 gate/up/down 拆分、dead-neuron、boundary/EHS
  等专用指标留在对应算法/优化器批次迁移。

## KL Aux

KL 继续使用官方 TopK 路由、梯度缩放、token 归一化和训练循环。实验侧通过
`MoESubmodules.router` 替换 Aux 公式，通过 `finalize_model_grads_func` 在完整步
结束后提交状态。与 YuLan 一样，直接通过 `--moe-router-load-balancing-type`
选择 `sym_kl_aux_loss`（local）或 `global_sym_kl_aux_loss`（global），
系数继续使用 `--moe-aux-loss-coeff`。默认仍为官方 Aux，不启用 KL。
实验入口扩展官方参数的 choices；内部运行参数副本适配官方 Aux 挂载接口，供模型和
数据 helper 复用。用户直接选择 KL，模型配置中的 `kl_aux` 保存真实算法及反馈选项。
不改 Core 或官方入口。
restart/容错检查通过通用可选参数复用已解析的 args，避免再次用官方基础 parser
解析实验选项；未传 args 的调用仍保留原行为。

固定 beta 的最小例子：

```bash
bash experiments/xcx/configs/baseline.sh \
    --disable-bias-linear --num-experts 4 --moe-router-topk 2 \
    --moe-ffn-hidden-size 128 --moe-aux-loss-coeff 0.01 \
    --moe-router-load-balancing-type sym_kl_aux_loss \
    --moe-router-sym-kl-loss-beta 0.7 \
    --async-log-writer --tensorboard-dir /tmp/xcx-kl-local
```

按以下顺序增加功能即可，beta 参数沿用 YuLan 名字：

| 层次 | 参数 | 行为 |
| --- | --- | --- |
| 固定 beta | `--moe-router-load-balancing-type sym_kl_aux_loss --moe-router-sym-kl-loss-beta 0.7` | TP/CP 域内的当前 microbatch 硬负载反馈 |
| Global | `--moe-router-load-balancing-type global_sym_kl_aux_loss` | TP/DP/CP 域内聚合，默认 `--moe-router-global-aux-mode microbatch` |
| 完整步反馈 | 再加 `--moe-router-global-aux-mode previous_step` | 使用上一完整步的冻结分布；首步由当前 microbatch bootstrap |
| 动态 beta | 再加 `--moe-router-sym-kl-loss-dynamic-beta` | 每个 MoE 层在完整步结束后独立更新 beta |

动态模式使用 `--moe-router-sym-kl-loss-beta-init`（默认 0.5），固定的 `...-beta`
参数此时不参与更新。默认范围 `--moe-router-sym-kl-loss-beta-range 0.25,0.75`，
步长 `...-beta-update-rate 0.001`，deadband `...-beta-deadband 1.0`。
每层用实际路由计数计算 `max(load) + min(load) - 2 * mean(load)`：大于 deadband
时增加 beta，小于负 deadband 时减少，最后截断到范围内。deadband 单位为原始路由 token 数。

还保留了 `--moe-router-sym-kl-loss-rho`、`...-coeff-clip`、
`...-tolerance LOWER UPPER` 和 `...-multiplicative-ste`。`legacy_prefix` 模式仅用于
重现历史等长 microbatch 实验；新的完整步反馈使用 `previous_step`。

- 数学实现和配置在 `losses/kl_aux.py`，Router 与各层状态在 `routers/kl_aux.py`，
  模型 spec 适配在 `models/gpt.py`，finalize/logger 在 `training/kl_aux.py`。
  KL 算法保留在实验目录；主体仅提供通用 writer setter 和上面的启动参数复用接口。
- 梯度使用 detached KL 系数乘 soft scores 的代理；日志展示 beta 加权的真实
  `KL(load || uniform)` 与 `KL(uniform || load)`。使用 scores proxy；logits、
  top-k boundary 等其他梯度代理尚未迁入。
- 保留 native `load_balancing_loss` 输出，并额外写入
  `sym_kl_load_balancing_loss` / `global_sym_kl_load_balancing_loss`、`sym_kl_beta`
  和对应的 `layer_0` 等分层 tag。KL 指标对应 forward 使用的反馈分布和 beta；
  beta tag 记录本步更新后的值。自定义 tag 采用 `--log-metrics-interval`。
- 普通 loss 模式补偿全局反馈的 DP 梯度平均；per-token 模式直接走官方 token
  归一化。两进程测试对照了不同 rank token 数和 padding 下的单批参考梯度。
- beta 作为 FP32 buffer 保存到官方 checkpoint，恢复时校验范围并更新 CPU cache。
  冻结反馈分布与步内统计沿用旧实现的非持久语义，恢复后重新 bootstrap，
  因此不承诺恢复后的反馈轨迹与不中断训练完全相同。日志步号按 checkpoint 起点校准。
- 当前使用官方 GPT MoE spec，支持 softmax/sigmoid。自定义 `--spec`、ModelOpt、
  CUDA graphs、MTP、hash 路由、router fusion、GTP 权重重物化，以及 global 模式的
  runtime Dynamic CP 尚未接入，会明确报错。

## EAR / EARH

基础实现使用 `probe × W_gate` 生成每个 expert 的路由行；EARH 在动态行上叠加
官方静态 Router weight。仍由官方 TopKRouter 完成选路、padding、Aux、observation
和 dispatch。先运行 EAR：

```bash
bash experiments/xcx/configs/baseline.sh \
    --disable-bias-linear --swiglu --num-experts 4 --moe-router-topk 2 \
    --moe-ffn-hidden-size 128 --moe-aux-loss-coeff 0.01 \
    --use-weight-aware-router
```

在此基础上追加 `--moe-router-add-static-router` 即为 EARH；追加
`--moe-router-load-balancing-type sym_kl_aux_loss --moe-router-sym-kl-loss-beta 0.7`
接固定 beta KL，或追加
`--moe-router-load-balancing-type global_sym_kl_aux_loss --moe-router-global-aux-mode previous_step
--moe-router-sym-kl-loss-dynamic-beta` 接完整步反馈和动态 beta。
KL 与 EAR 使用同一份 spec，不新增训练循环或配置格式。

梯度模式沿用 `--moe-router-grad-mode`，只控制路由分支，expert 自身的主任务训练继续正常进行：

| 模式 | 动态路由分支的梯度 |
| --- | --- |
| `end2end`（默认） | probe、expert gate、可选 channel beta 均参与训练 |
| `probe_only` | detach expert gate，训练 probe 和可选 channel beta |
| `gate_only` | 固定 probe 和 channel beta，训练 expert gate |
| `stop_w_gate` | detach 整个动态行；输入和 EARH 静态 Router 仍可获得梯度 |

- `--moe-router-per-expert-probe` 使用每 expert 的独立 probe；默认共享。
  `--moe-router-probe-init mean|ones|zeros` 保留旧初始化，默认 `mean` 为 `1 / FFN`。
- `--moe-router-ear-normalize` 仅归一化动态路由行，再乘
  `--moe-router-ear-normalize-tau`（默认 1）；EARH 的静态行在归一化之后相加。
  `--moe-router-use-channel-beta` 在归一化之前增加可训练的 hidden-channel 缩放。
  `stop_w_gate` 下同时冻结 channel beta，避免旧实现中它被 detach 后成为无梯度参数。
- EARH 的 `--moe-router-lb-static-only` 保留融合 logits 的选路和负载反馈，
  仅让 balancing 梯度进入静态 weight/bias，阻断该项到输入、probe、expert gate
  和 channel beta 的梯度；主任务及 native z-loss 保持原路径。
  已覆盖官方 Aux、seq/global Aux 和本目录的 local/global KL。
- `routers/ear.py` 只计算 logits、管理梯度路径；`models/moe.py` 从官方 experts
  读取可微 gate 权重视图，并由薄的 MoELayer 子类绑定 provider。
  Router 不重复注册 expert 参数、不缓存跨 forward 的权重/计算图，也不改变参数的 checkpoint 层级。
  EP 使用可微 all-gather；probe/channel beta 使用官方梯度同步和 checkpoint。
- 参数日志增加 `moe::router::probe_vector` 和 `moe::router::channel_beta`；
  `moe::router::weight` 记录 EARH 的静态参数。纯 EAR 的动态行不是独立参数，
  本批不为它伪造参数日志。logits/负载指标继续复用基础 logger。
- 当前要求 gated experts（如 `--swiglu`）、`--expert-tensor-parallel-size 1`、
  `--no-gradient-accumulation-fusion` 和普通、完整物化的 FP32/BF16/FP16 权重。
  不支持 FSDP、overlap param gather、MoE overlap、latent experts、FP8/FP4，
  以及上文列出的自定义 spec/ModelOpt/graphs/MTP/hash/router fusion/GTP 重物化。
  这些组合会在构建前报错，避免改动 Core 的权重和梯度管理。
  本机实测的是 local SequentialMLP；TE grouped 布局仅做了权重视图/梯度单测，尚未实机验证。

### EARH 梯度流与 aux-free bias

四种模式只控制路由分支，不冻结 expert 自身的 FFN 主任务训练；EARH 的静态 router
始终保留主任务梯度：

| `--moe-router-grad-mode` | probe | 路由分支 → expert gate | expert 自身 FFN |
| --- | --- | --- | --- |
| `end2end` | 训练 | 有梯度 | 训练 |
| `probe_only` | 训练 | 截断 | 训练 |
| `gate_only` | 固定初始化值 | 有梯度 | 训练 |
| `stop_w_gate` | 固定初始化值 | 截断 | 训练 |

例如检查 detach-gate 干预，沿用官方 global Aux：

```bash
earh_args=(--disable-bias-linear --swiglu --num-experts 4 --moe-router-topk 2
           --moe-ffn-hidden-size 128 --use-weight-aware-router --moe-router-add-static-router)
bash experiments/xcx/configs/baseline.sh "${earh_args[@]}" \
    --moe-router-grad-mode probe_only \
    --moe-router-load-balancing-type global_aux_loss --moe-aux-loss-coeff 0.005
```

改为 `gate_only` 检查 freeze-probe；`--moe-router-lb-static-only` 则只隔离负载均衡
梯度，让它回传静态 router，保留动态路由分支的主任务梯度。

aux-free bias 直接复用官方 sign 更新，无需新增 Router 或实验开关：

```bash
bash experiments/xcx/configs/baseline.sh "${earh_args[@]}" \
    --moe-router-grad-mode end2end \
    --moe-router-load-balancing-type none --moe-aux-loss-coeff 0 --moe-z-loss-coeff 0 \
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 0.001
```

bias 只影响 expert 选择，combine weights 使用未加 bias 的 sigmoid scores。
bias 是 FP32 buffer，通过官方 finalize 按完整步的全局 token 计数更新；它不参与反向传播，
并随官方 checkpoint 保存。其余三种梯度模式也可与 bias 组合。
对比有无 bias 时保留相同的 sigmoid、归一化、seed 和训练配方；无 bias 对照仅去掉
`--moe-router-enable-expert-bias`。源 YuLan rebuttal 的 sign 模式已有对应路径，
`rms_norm` bias update 尚未迁入。

这里验证的是梯度路径消融和 bias 功能。旧版 LM/LB 梯度范数/夹角分解 logger 尚未迁移。
此外，官方 `global_aux_loss` 使用步内累计前缀负载；当前 YuLan 默认使用当前 microbatch
反馈，源端还支持 previous-step。现有 `--moe-router-global-aux-mode` 只接入 KL，
没有覆盖官方 global Aux；复现旧 EARH 的完整收敛曲线前仍需对齐该反馈语义和梯度缩放，
并完成 TE/Muon 与真实训练配方验证。上面的 baseline 是 mock data 功能检查。

## EAR gate-up / 初始化 / 正交 loss 和 update

继续使用旧参数名，默认都关闭，不增加配置格式。基础 EAR 参数后可逐项追加：

```bash
# 非线性 gate-up 路由；需要 --swiglu
--moe-router-gate-up-interaction --moe-router-gate-up-interaction-chunk-size 2048

# 特殊 expert gate 初始化
--moe-router-w-gate-init mean_orth --moe-router-w-gate-init-seed 12345

# 普通 EAR/EARH 的正交目标，进入正常 backward/optimizer
--moe-router-orthogonal-loss-coeff 0.05 \
--moe-router-orthogonal-update-mode coupled \
--moe-router-orthogonal-preprocess none

# 或：与 Adam moments 分离的直接参数更新
--moe-router-orthogonal-loss-coeff 0.05 \
--moe-router-orthogonal-update-mode decoupled
```

gate-up 使用 `alphaᵀ [SiLU(W_gate x) ⊙ (W_up x)]`，共享/独立 probe、EARH 和
static-only balancing 继续可用，选路、padding、KL/Aux、dispatch 均走现有接口。
chunk 参数控制单次投影的 token 数，反向仍需保存各 chunk 的计算图。
四种梯度模式保持旧语义；其中 `stop_w_gate` 在这个非线性模式下只 detach gate/up 权重，
probe 固定，输入仍有经过 SiLU/乘积的梯度。它与线性 EAR detach 动态行的含义有所不同。
当前不与 row normalization、channel beta、reps 编译/prefetch 或正交目标组合；
旧 gate-up 分支未实际应用正交 loss，这里明确报错，避免静默忽略参数。

EP gate-up 修正了旧代码直接拼接各 rank 本地 logits、从而混用其他 rank 输入的问题：
先可微 gather 输入，由本地 experts 评分，再 gather 得分并取回本 rank 对应的行。
通过 padding 支持不同 rank 的 token 数；输入、gate/up 和 probe 梯度都保持可微。
不 gather 整套 expert 权重，但增加了输入/得分通信；本批验证正确性，不宣称其吞吐收益。
EP=1 的输出和梯度与旧方法精确对齐，EP>1 对照完整 expert 矩阵的数学参考。

`mean_orth` 用 CPU 独立 generator + QR 生成全局 expert 的行正交均值方向，
叠加 FFN 维零均值噪声，保留原来的 std 和按层/EP offset 派生 seed 的规则。
仅改 gate 半部，并将 probe 置为 `1 / FFN`（会覆盖 `probe-init ones/zeros`）；
up/down 和默认 expert 初始化保持原样。要求 experts <= hidden、ETP=1。
在获得全局 layer_number 后只执行一次，`perform_initialization=False` 时跳过，
checkpoint 恢复后不会再次初始化。不同 EP 划分共享均值方向，但噪声保留旧分片 seed 语义，
因此不保证改变 EP 大小时得到相同的完整初始化。

正交目标位于 `losses/orthogonal.py`：取 channel beta 之后、row normalization 和
EARH 静态相加之前的动态行，转 FP32，按行 L2 normalize，然后计算
`mean((R @ R.T - I)^2)`。`--moe-router-orthogonal-preprocess` 支持
`none/mean_center/pca`，PCA 由 `--moe-router-orthogonal-pca-rank`（默认 1）控制，
rank=0 等价于只中心化。保留可微 SVD 及其在重复奇异值附近的梯度敏感性。

- `coupled` 用官方 `MoEAuxLossAutoScaler` 注入目标，系数为用户值 / TP-CP group size。
  为对齐 YuLan，per-token loss 下仍乘 `logits.shape[0]`（序列长度），没有改成有效 token 计数；
  batch>1 或 padding 时这与 native Aux 的 token 口径不同。
- `decoupled` 由 `training/orthogonal.py` 的官方 Callback 在成功的普通 optimizer step
  之后重新计算目标，按该步实际使用的参数组 LR 执行 `p -= lr * orth_grad`。
  LR 在 step start 保存，避免误用 scheduler 已推进后的 LR；跳过更新时不执行。
  不写正常梯度缓冲或 Adam moments，也不额外执行 clipping/weight decay。
- 更新映射到真实 FC1 参数及 FP32 master，再同步回 BF16 model；不再把打包后的
  非参数 tensor 交给 optimizer。EP 下补偿重复目标的 SUM，并归并共享 probe/channel beta，
  使每个全局正交目标只更新一次。这两点修正了旧 decoupled 路径 expert 更新未生效及 EP 不一致的问题。
- decoupled 首批支持 TP=CP=1、reference reps、普通复制参数的 optimizer；
  Distributed/layer-wise optimizer、precision-aware optimizer 及 FSDP 尚未接入，会报错。
  实测为 FP32/BF16、local backend；TE/FP16 尚未实机验证。`stop_w_gate` 不做独立正交更新。
- `router_orthogonal_loss` 及分层 tag 通过现有 writer 输出，支持 async writer 和恢复后的步号。
  正交指标记录 forward 使用的目标；decoupled 更新在 Adam 后重新计算，二者时点不同。
  实验入口自动注册 callbacks；程序化训练需显式传入 `EAROrthogonalUpdate`（以及需要时的 logger）。
  callback 没有新增持久状态，checkpoint 复用官方 model/master/moments。

## EAR packing / compiled reps / prefetch

只增加一个参数 `--ear-reps-mode`，默认 `reference`，沿用上面的 EAR/EARH 和 KL
参数。先用同一配置比较数值与梯度，再选择是否开启优化：

| 模式 | 行为 |
| --- | --- |
| `reference` | 本目录的基础 EAR，保持原逐层 BMM 和原 representation gather |
| `local_pack` | 编译该层 gate-only packing，投影和通信保持原路径 |
| `local_fused` | 一起编译该层 packing 和原尺寸 BMM，不增加跨层调度 |
| `buffered` | 逐层 contiguous all-gather / SUM reduce-scatter，作为通信布局对照 |
| `prefetch` | 在独立 stream 提前准备下一 EAR 层的路由行，使用 event 交接 |
| `batched` | 保留逐层 BMM，将同一 block 的路由行合并为一次 gather / reduce-scatter |
| `compiled_stack` | 在 `batched` 基础上，用一个图编译各层 packing 和各自的原尺寸 BMM |
| `compiled_stack_async` | 在独立 stream 准备 `compiled_stack`，首个 Router 消费时等待 event |

例如在 EAR 启动参数后追加 `--ear-reps-mode local_fused`，或在满足下述范围时追加
`--ear-reps-mode compiled_stack`。基础版本原已只取 gate 行，`local_pack` 不再宣称
减少一半权重拷贝；它与 `local_fused` 的收益需要实际测量。

- `routers/ear_compiled_reps.py` 只有三个编译入口：packing、单层 reps、跨层 reps。
  继续使用明确的 FC1 权重布局，不恢复旧代码按尺寸猜测转置方向的启发式。
  probe 的 dtype 转换留在编译边界外；不把多个 BMM 合成曾出现舍入差异的大 BMM。
  只返回实际消费的 reps，不保留无用的可微 gate 输出。
- `models/ear_reps.py` 负责一次 forward 内的调度。通过官方 `post_wrap_hooks`
  安装模块 hooks，不改 TransformerBlock、MoELayer forward 或训练循环。
  每次 forward 重算路由行，进入时等待之前主 stream 的权重更新；消费后移除 pending 引用，
  正常返回和异常都清理步内状态。不存在跨 microbatch/optimizer step 的 reps 缓存。
- reps 使用成员及顺序与 EP 相同的独立 communicator，与 token dispatch 分开。
  空 tensor 的 autograd 依赖约束反向 collective 顺序；反向仍是 SUM reduce-scatter，
  不 detach expert/probe 的训练梯度。关闭 runtime 时移除本批 hooks、同步并释放 side stream/group。
- `local_pack/local_fused` 保留四种梯度模式，可与 full recompute 组合。
  其余调度模式当前要求 `end2end`、TP=PP=1、统一 expert 形状、无 activation recompute、
  无 DDP overlap-grad-reduce/overlap-param-gather，并继续遵守 EAR 的 ETP=1/FSDP/精度约束。
  普通 Megatron DDP、EARH、KL、static-only、异步日志和 checkpoint 可以组合。
  TE 布局只有适配器检查，实机验证使用 local SequentialMLP。
- 实验入口自动管理 runtime 生命周期。程序化构建时，使用
  `with configure_ear_model(...):` 包住 `build_distributed_models` 和训练；直接调用
  `build_model` 时需要显式 `EARRepsRuntime.install`。缺少安装会报错，不会静默退回普通模式。

可复用的检查与性能入口：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1 \
python -m torch.distributed.run --standalone --nproc-per-node 2 \
    --module experiments.xcx.tests.benchmark_ear_reps --ddp \
    --output /tmp/ear-reps-results.json
```

使用 `--check-only` 只做严格对照；`--hybrid --per-expert-probe --channel-beta
--moe-router-load-balancing-type global_sym_kl_aux_loss --static-only`
检查 EARH/KL 组合。每种模式先比较输出、输入梯度和所有参数梯度
（rtol=atol=0），重复执行、连续累积 3 个 microbatch，再修改参数后重新对照。
各 rank 使用不同输入种子。任何 rank 比较失败都会在所有 rank 上结束，避免部分 rank 继续通信。
GPU 编译函数还在官方 determinism manifest 登记了位级重放测试。

性能测量使用官方 GPT decoder（包含 attention/MoE/输出 head）与可选的 Megatron DDP，
通过 decoder input 输入以检查输入梯度；不含 embedding、数据加载或 optimizer step。
编译和正确性检查不计时，每轮 warmup 后测量；奇偶轮反转模式顺序。
每个样本记录各 rank 的 wall time、峰值 allocated/reserved，吞吐按最慢 rank 计算。
JSON 保留所有原始样本、编译环境、硬件和 CPU affinity，不删除异常长尾。

本机测量（2026-11-11）：A800-SXM4-80GB、PyTorch 2.10.0+cu130，EP=2、BF16、
8 层、hidden/FFN=256、8 experts/top-2、每 rank 256 tokens，累积 2 个 microbatch，
普通 Megatron DDP、local SequentialMLP。每模式 4 轮，每轮 warmup 5 次、测 15 次；
进程绑定 CPU 40–47，计时期间没有并行运行其他自有测试，机器上仍有其他用户作业。
每轮 warmup 后清空空闲 allocator cache 并复位峰值，首个样本可包含重新分配开销。

| 模式 | 总耗时中位数 ms | 吞吐 tokens/s | 峰值 allocated MiB | 峰值 reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| reference | 192.993 | 5306 | 1179.2 | 1194 |
| local_pack | 196.019 | 5224 | 1179.2 | 1194 |
| local_fused | 195.981 | 5225 | 1179.2 | 1194 |
| buffered | 193.080 | 5304 | 1179.2 | 1194 |
| prefetch | 196.692 | 5206 | 1179.2 | 1196 |
| batched | 186.736 | 5484 | 1115.2 | 1130 |
| compiled_stack | 184.458 | 5551 | 1115.2 | 1130 |
| compiled_stack_async | 185.992 | 5506 | 1115.2 | 1136 |

本配置的 `compiled_stack` 吞吐中位数比 reference 高约 4.6%，峰值 allocated 少 64 MiB。
四轮 reference 中位数为 192.14–194.29 ms，compiled_stack 为 184.01–184.85 ms。
局部 packing/编译和逐层 prefetch 没有加速。原始数据存在长尾：reference 均值/标准差为
209.622 ± 95.226 ms，compiled_stack 为 186.074 ± 6.464 ms；prefetch 与 async 模式的
最大样本分别为 2395.29/1114.45 ms，均已保留。这些数值只说明当前形状、软件和后端的
forward/backward 结果，不外推为 TE、大模型或含 optimizer 的完整训练收益。
显存为 PyTorch allocator 统计，不包含全部 CUDA context/NCCL 外部分配。
本次原始文件为 `/tmp/xcx-ear-reps-a800-ep2-distinct.json`；更换实际训练形状后应重新运行检查与测量。

## 文件放置与接入约定

- `training/pretrain_gpt.py`：只组装参数、配置和训练调用；batch、LM loss 和
  forward step 复用根目录 `pretrain_gpt.py`，训练循环复用 `megatron.training.pretrain`。
- `configs/`：直接使用 Megatron 参数的启动脚本，暂不增加配置格式。
- `routers/`、`losses/`、`models/`、`logging/` 各目录直接放对应实现，
  不增加注册框架；配置和算法实现按实际需求扩展。
- Router 通过 `MoESubmodules.router` 接入；EAR 的 expert 权重访问放在自定义
  MoELayer 中。指标优先使用现有 tensor observation 和 callbacks。
- KL 的 step 状态更新需要核对官方 finalize/reset 的顺序，通过模型配置的
  `finalize_model_grads_func` 接入。保持官方梯度同步和归一化语义。
- 依赖方向为 `experiments.xcx -> megatron`。避免全局 monkey patch；确实缺少
  扩展点时，再单独评估最小的主体代码改动。

## 分批迁移

| 批次 | 内容 | 主要验证 |
| --- | --- | --- |
| 0 | 本目录与官方 GPT baseline 入口 | 短训练、checkpoint 保存与恢复 |
| 1 | AsyncSummaryWriter、AsyncWandbWriter | 顺序、数据快照、flush/close、后台异常 |
| 2 | 基础指标采集与输出 | 指标名、跨 rank 聚合、重计算去重、关闭开销 |
| 3 | KL Aux：固定 beta、global feedback、动态 beta | loss/梯度、token 归一化、step 更新与恢复 |
| 4 | EAR、EARH 和梯度模式 | 路由结果、probe/expert 梯度、EP 与 KL 组合 |
| 5 | packing、compiled reps、prefetch 与配套工具 | 数值对齐、吞吐和显存 |
| 6 | gate-up、mean_orth、coupled/decoupled 正交目标 | 源公式/梯度、EP 数学参考、master/moments、恢复 |

迁移对照起点（2026-11-11）：

- 目标：Megatron-LM `685fb8bd970185e9ac3283fe32a5d6667bf4b268`，MCore 0.20.0。
- 来源：相邻 YuLan-Pretrain 工作区，HEAD 为
  `def5024ab70be8cc2f55d2c4e2459697bf9d8c32`，MCore 0.16.0rc0。
  来源还包含未提交修改和未跟踪实现，迁移每个功能前重新记录相关 diff 与文件。
- 当前完成 baseline、async writer、基础指标 logger、固定 beta / global feedback /
  动态 beta 的 scores-proxy KL Aux、基础 EAR/EARH、梯度模式和 KL 组合，
  以及可选的 packing、compiled reps、跨层 batching/prefetch、gate-up、mean_orth 和正交 loss/update。
  其他梯度代理、专用诊断和性能优化尚未迁入。

## 剩余迁移清单与下一批优先级

盘点日期：2026-11-11。范围是 YuLan 当前 `new_megatron/moe_router_xcx` 工作区：
包含 HEAD 中已提交的自定义实现、tracked diff 和未跟踪源码；不只看 `git diff HEAD`，
也不把其它分支、benchmark 输出、checkpoint、图表和 notebook 副本算成独立待迁功能。
下表的“目标原生”指当前 Megatron-LM 0.20 checkout 中已有代码，不代表已经做过 YuLan 数值对齐。

已迁主线：基础/异步日志、scores-proxy KL（固定/global/动态 beta）、EAR/EARH、梯度模式、
packing/compiled reps/prefetch、gate-up、mean_orth 和正交 loss/update。
以下是实际缺口；优先级按当前 EAR/KL 实验的关联、依赖和侵入成本排序，不按旧提交时间排序。

| 优先级 | 尚未迁入或只迁了一部分 | 来源线索 | 轻量接入方式与边界 |
| --- | --- | --- | --- |
| P0 / EARH 复现前置 | 普通 global Aux 的反馈模式与梯度缩放对齐 | 源 `_global_aux_counts` / `_apply_global_aux_loss`；目标官方同名方法 | 源默认 microbatch，目标仍为步内 prefix；现有反馈参数只接 KL。先在实验 Router 接普通 global Aux，复用已有完整步状态管理，不改 Core |
| P0 / 下一批 | `global_centered_aux_loss`；KL/原 global Aux 的 `logits` proxy | `moe_utils.py::centered_load_balancing_loss_func`、`logits_aux_scores`；`router.py::_apply_global_centered_aux_loss`、`_aux_loss_proxy_scores` | 只扩展实验侧 loss/Router/spec，复用完整步反馈和 finalize；不复制旧 routing/训练循环 |
| P0 / 随后一小批 | KL 的 `topk_boundary` proxy、temperature | `moe_utils.py::_TopKBoundaryProxy`、`topk_boundary_scores` | 单独保留隐式阈值求解与自定义 backward，先 gradcheck/梯度参考；不与真实选路混为一件事 |
| P1 | 只读 boundary 指标的余项 | `router_boundary_metrics.py` | 复用 observation；基础 logger 已有 top-k margin，目标 TensorMetrics 已有 router-health，补的是 tie、logit-std 归一化和 alpha 阈值比例等不同口径 |
| P1 | 逐 expert gate/up/down 参数及梯度统计；weighted/unweighted activation、dead-neuron | `experts.py` 中 `_save_expert_param_stats`、`_save_unweighted_activation_stats`、`_save_dead_neuron_stats`；`moe_utils.py::track_moe_router_expert_metrics` | 延续现有 callback/指标文件，使用明确权重视图和观测点；TE 未暴露原始激活时明确限制，不为日志复制 fused forward |
| P1 | EAR checkpoint → HF、HF 推理参考和离线对齐 | `scripts/distributed_checkpoints_convertor/conver_EAR_ckpt.sh`、`utils/parse-moe-vars-from-mcore-config.py`、`tasks/EAR_ANALYSIS/ear_hf.py` | 用目标 checkpoint API 读参数，转换器留在工具侧；线性 EAR 的折叠须保留 dtype 边界，gate-up 需要非线性 HF 参考，不能当静态线性 Router 导出 |
| P1 / 验证而非重迁 | 真实训练配方、TE grouped + 目标 Muon、现有功能的组合范围 | `scripts/pretrain/moe_pretrain_template.sh`、`scripts/pretrain/run/EARpaper/`；本目录各批验证限制 | 只精选常用 EAR/EARH/KL 配方到 `configs/`，其余用目标 CLI；单列参数分组、split、LR/master、checkpoint 验证。目前主要实测 local backend/Adam，TE 和 emerging_optimizers 依赖环境未就绪 |
| P2 | KERN、SigmoidAbs Router；top-k combine normalization / raw scores / sqrt；bias 的 RMS update | `router.py::SigmoidAbsRouter`、`moe_utils.py::topk_routing_with_score_function`、`get_updated_expert_bias(method='rms_norm')` | 先小 Router 子类和实验侧概率后处理；这些会改变主任务的选路/权重，不与仅改 Aux 梯度的 proxy 同批。目标已具备普通 aux-free expert bias，缺的是自定义变体 |
| P2 | ERC loss、EHS / pairwise expert similarity | `moe_layer.py::_compute_erc_loss`；`router.py::_log_ehs_stats`；`experts.py::compute_local_expert_outputs_for_similarity` | ERC 是训练目标，EHS 是诊断，分批处理。源 EAR 的 `get_router_weight_matrix_for_loss()` 返回 None，不能把 ERC 直接当成既有 EAR 目标。反事实 expert 输出采样另评估开销 |
| P2 | Routerless AoE（full/low rank）、Routing-Free、自适应 lambda、首批 bias 校准 | `router.py::RouterlessAoERouter`、`RoutingFreeRouter`；`experts.py::AoEGroupedMLP` | 先 full-rank 评分，再低秩 expert，再变活跃数及跨层 density/lambda；复用 MoESubmodules/dispatcher/finalize。变 token/expert 数和跨层状态比普通 TopK 更复杂 |
| P2 | YuLan 的 soft/hard GLU clamp、final-logit clamp、expert output norm；split/spectral 初始化差异 | `activations.py`、`fused_bias_swiglu.py`、`mlp.py`、`experts.py`、`core/utils.py` | 先核对目标现有激活/初始化接口；仅把缺失语义放到实验侧 activation/spec/init hook。不能按近似名称直接替换，见下文 |
| P3 | 自定义 Muon 拆分/向量化/缩放、MuonBall、SpectralBall、`MupAdamW`、update/grad RMS | `optimizer/muon.py`、`muon_ball_optimizer.py`、`spectral_ball_optimizer.py`、`mup_adamw.py` | 基础 Muon/MuP 优先走目标原生；球面优化器依赖外部 emerging_optimizers，先核对依赖版本/registry，再补薄适配，不整包覆盖 optimizer/distributed optimizer |
| P3 | LM/LB 梯度分解、范数/夹角、异常梯度诊断 | `router_gradient_diagnostics.py`、`training/grad_monitor.py` | 源诊断仅支持普通 Router 的有限范围，且会修改 TE backward context、临时替换清理函数并抑制 Aux backward。先设计可验证的局部观测方式，不照搬全局 monkey patch |
| P3 | 其它模型/训练工程扩展 | token-shift/conv、gated hidden states、embedding deviation/dropout、冻结 LN；定制层映射/upcycling、benchmark eval、Parquet/zst/glob 预处理、job UUID、TB→WandB 同步 | 模型用独立 spec/builder，数据/评估/导出/运维用独立脚本或 callbacks；先比对目标已有实现。旧机器路径和整套 launcher 不并入训练主体 |

目标版本已有、应优先复用的内容：

- **QB**：目标 `TopKRouter.quantile_balancing`、`qb_beta`、EMA、完整步 finalize 更新均已存在。
  应做旧参数/行为/恢复对齐，避免再写一个 QB Router。
- **普通 Muon 与 MuP**：目标有 `emerging_optimizers` 接入、Muon 分布式路径，以及
  `get_mup_config_overrides`。这不等于已覆盖 YuLan 的全部 split/vectorize/球面变体。
  源 `MupAdamW` 实际在 Adam 前缩放梯度，也不能直接当成目标 LR/epsilon override 的别名。
- **激活原语**：目标有 hard clamp、`tanh_soft_clamp`、SiTU/SiTU-GLU。
  源 soft-GLU 是先 clamp gate 再 SiLU；SiTU 则保留原 gate 的 sigmoid，二者不等价。
  final-logit clamp、训练/推理一致性和精确梯度仍需单独核对。
- **模型/观测基础设施**：目标已有 latent MoE、attention output gate、GDN 等 attention 路径、
  TensorMetrics/router diagnostics、通用 profiler 和标准 aux-free bias。
  功能存在不等于本实验入口已经接入或与旧脚本默认值一致，应补 adapter/配方/验证而非重写 kernel。
- **旧性能补丁**：单 rank expert-sort 快路、all-to-all trace 等先与 0.20 dispatcher/profiler 对照，
  只在仍有差异且基准证明收益时考虑改动；不因旧分支有补丁就回填 Core。

EARH 复现检查补充（2026-11-12）：先完成普通 global Aux 反馈与梯度缩放对齐，
再迁 **global_centered Aux + logits proxy**：

1. 对齐公式、梯度和日志含义，保留 centered 目标“没有前导 E 倍数”的尺度。
   centered 不是“再去一次均值的同名 Switch”；在归一化 scores 下可能有梯度关系，但
   不同系数、sigmoid/STE 代理与反馈模式不能据此视作同一目标。源 centered 路径也不支持 logits proxy，
   第一批保留该组合限制，logits 先接 KL 和原 global Aux。
2. 复用已有 global feedback 的 microbatch/previous-step 接口，核对梯度平均、per-token、
   padding、不同 rank token 数、static-only 和恢复 bootstrap；不另造 step 状态框架。
3. 证明只更换 Aux 代理时 logits、top-k assignments、combine weights 和 dispatch 不变，
   以解析梯度和单批参考核对 local/global 路径；沿用现有入口和日志 writer。
4. `topk_boundary` 随后单独接入：它包含根求解和隐式梯度，不是一个只读 margin 指标。
   先过 gradcheck，再验证 EARH/static-only 与 KL 的组合。

这一批不混入新 optimizer、AoE/Routing-Free 或 retained-graph 梯度诊断；
代码预计只落在现有 `losses/`、`routers/`、模型参数接入和测试中，优先保持 Core 零新增修改。
本清单是审计与排期，没有提前迁入上述新算法。
历史日志中还能看到 checkpoint retry / `SAVE_CHECKPOINT_NOW` 等提交，但当前工作区未找到
对应实现，未将历史提交标题直接计为现存待迁功能。

## 本批验证

- 已通过 Python 语法、isort 和 Bash 语法检查。
- 已在 A800 单卡、local backend、FP32 下完成 4 步训练，保存 torch_dist
  checkpoint，再从第 4 步恢复到第 6 步；没有 NaN 或跳过更新。
- 本机系统 C++ 运行库较旧，验证时仅对启动进程设置
  `LD_PRELOAD=/usr/local/conda/lib/libstdc++.so.6`；脚本不包含机器专用路径。
- 当前上游 `--help` 会因 `mhc_recompute_layer_num` 文档中的未转义 `%`
  触发 argparse 格式化错误，普通训练参数解析正常；本批未改动该上游问题。
- Async writer 的 13 项 CPU 回归检查通过，包含真实 TensorBoard event 读写；
  WandB 使用替身验证接口、数据快照和退出行为，尚未验证真实 WandB 服务。
- 开启异步 TensorBoard 的 A800 单卡 4 步训练通过，event 文件中的四步 loss
  与训练输出一致，训练 loss 与 baseline 一致。
- 基础指标的 15 项 CPU 回归检查通过，覆盖数值公式、TP 合并、不同 microbatch
  长度、padding、PP 空记录、恢复后的 interval、重算去重、异常清理，以及输出/梯度不变。
- A800 双卡 4 步短训练通过：dense TP=2、dense TP=2 + full recompute、MoE DP=2、
  MoE PP=2。读取 TensorBoard 验证所有新增序列只在第 2/4 步出现，数值有限，
  PP 两阶段的全局层号齐全。TP=2 普通训练与 full recompute 的四步 loss 一致。
  这些是 local backend、FP32 的功能检查；MoE 使用 `--disable-bias-linear`。
- KL 与 YuLan 原函数的 96 组 loss、KL 指标和梯度对照通过。全部实验侧单进程
  回归为 61 项通过；另外 4 项双进程 CPU 梯度测试通过，覆盖普通/per-token loss、
  不同 rank token 数和 padding。
- KL 短训练通过：固定 beta 单卡；global previous-step + DP=2 + per-token loss；
  动态 beta + MoE/dense 混合 PP=2（包含无 MoE 的日志阶段），checkpoint 保存并恢复到
  第 6 步；动态 beta + DP=2 + BF16 + full recompute。均为 local backend 功能验证。
  已读取 checkpoint 确认 beta 为 FP32，并核对恢复后的 KL/基础指标步号。
- EAR/EARH 与 YuLan 原方法的 64 组 logits/梯度对照逐项一致；覆盖四种梯度模式、
  共享/独立 probe、归一化、channel beta 和静态行组合。另有 8 项双进程 CPU
  EP gather/backward 对照，比较完整 expert 矩阵参考下的输入、gate、probe 和静态参数梯度。
  再有 4 项 EP + global KL 对照，覆盖不同 rank token 数、padding、普通/per-token
  归一化和 static-only，验证 expert 分片与普通 Router 参数的梯度缩放。
- 全部实验侧单进程回归为 122 项通过、16 项分布式检查跳过；分布式检查单独启动。
  EAR 单测还覆盖五种 expert 权重布局、static-only Aux/KL、padding、per-token loss、
  参数注册和 spec/config 序列化。
- A800 local backend 短训练通过：EAR 四种梯度模式；EAR + 固定 beta KL + per-token loss；
  EARH + EP=2 + global previous-step KL + 动态 beta + BF16 + full recompute + async logger。
  后者保存第 4 步 checkpoint 并恢复到第 6 步；读取 checkpoint 验证 probe 已更新、
  beta 保持 FP32，48 条 EAR/KL 日志序列数值有限，恢复后的步号为 5/6。
  本批没有新增任何 Core 或 training 主体代码修改。
- EAR 优化批次：144 项实验侧单进程检查通过、18 项分布式检查单独启动；
  双进程 EAR 14 项、KL 4 项通过。覆盖 packing 源权重梯度、四种局部梯度模式、
  step 生命周期、异常清理、参数修改、累积、独立 group 和缺少安装的报错。
  GPU 编译重放 6 项、官方 determinism manifest 263 项检查通过。
- 官方 GPT + Megatron DDP 的 EP=2/4 GPU 严格检查通过；八种模式比较输出、输入及
  所有参数梯度。EP=2 覆盖 BF16 共享 probe 和 FP32 EARH/local KL；EP=4 覆盖 BF16
  独立 probe + EARH/global KL/static-only。均检查重复 forward、三个 microbatch 累积和
  参数修改后的新路由行；计时任务先完成检查。
- GPU 短训练通过：`prefetch` + EP=2 + EARH + global KL/动态 beta；
  `compiled_stack_async` + EP=2 + EARH/static-only + global previous-step KL/动态 beta，
  保存第 4 步并恢复到第 6 步，48 条 EAR/KL 日志序列数值有限且恢复步号为 5/6。
  `local_fused` 的 probe_only/gate_only/stop_w_gate 与 BF16/full recompute/local KL 组合通过。
  优化默认关闭；本批主体代码未新增修改，仅按仓库要求登记编译函数的 determinism 测试。
- EAR 算法扩展批次：206 项单进程回归通过，34 项分布式检查单独启动；
  原 EAR/KL 的 18 项及新 gate-up/decoupled 的 16 项双进程检查通过。
  与 YuLan 原方法进行 96 组 gate-up 输出/梯度（FP32/FP64/BF16）、6 组正交 loss/梯度、
  6 组初始化精确对照。新增更新 oracle 覆盖不同参数组 LR、scheduler 时机、master 同步、
  Adam moments 不变、skip、全局层号初始化一次，以及 EP 不同输入/不等 token 数。
- GPU 短训练通过：gate-up + mean_orth + EARH/static-only + EP=2 + BF16/full recompute +
  global previous-step KL/动态 beta；coupled/PCA + EP=2 + BF16/full recompute + per-token/local KL；
  decoupled/mean_center + mean_orth + EP=2 + BF16 master + global KL/动态 beta + 线性 LR 调度。
  后者从第 4 步恢复到第 6 步，已核对正交指标的 5/6 步号和有限值。
  `coupled + compiled_stack_async + EARH/global KL` 另通过官方 GPT/DDP 的输出、输入及全部参数梯度
  零容差对照，包含重复、三 microbatch 累积和参数修改后的重算。本批未新增 Core/训练循环修改。

- KL CLI 统一后，KL/EAR/reps/advanced 的 189 项回归通过；补充启动参数复用测试后，
  KL 单文件 45 项通过。单卡固定 beta KL（含 expert capacity）和双卡 EP=2 + BF16 +
  EARH/static-only + global previous-step/dynamic beta 短训练通过；后者保存第 2 步并
  恢复到第 4 步，配置保存真实 KL 模式，6 条 KL/beta 日志在恢复后使用第 3/4 步。

- EARH 梯度流补充：单进程 74 项、双进程 18 项通过，覆盖主任务 expert/路由/Aux
  梯度分解参考、四种模式、static-only、bias 只参与选路、padding、全局计数更新及 reset。
  A800 local FP32 的 11 组两步训练通过（四种 global Aux、四种 aux-free、static-only、
  相同 sigmoid 的无 bias/Aux 对照）；checkpoint 确认冻结 probe 保持初值，静态 router
  与 expert gate 均有更新，bias 以 FP32 保存并更新。EP=2 + BF16 + full recompute +
  probe_only/aux-free 保存第 2 步并恢复到第 4 步，负载日志步号连续。本批仅新增测试和说明。

- Torch 2.6 兼容检查：真实 `2.6.0+cu124`、缺少 E8M0 dtype 的环境下，普通 MoE、
  EARH/global previous-step KL/动态 beta、EARH/probe_only/aux-free 均通过两步 BF16
  mock-data 训练（A800、local backend、Adam）；KL 配置还保存了第 2 步 checkpoint。
  新增能力检查测试在 2.6 下 6 项通过、1 项因缺少 dtype 跳过；2.10 下 7 项通过，
  232 项实验回归及 GPU E8M0 字节转换 replay 通过。未变更 FP8 数值 kernel。

异步日志的 CPU 检查不依赖官方 GPU 单测 fixtures，可单独运行：

```bash
python -m pytest -q experiments/xcx/tests/test_async_writer.py
python -m pytest -q experiments/xcx/tests/test_metrics.py
python -m pytest -q experiments/xcx/tests/test_kl_aux.py
python -m pytest -q experiments/xcx/tests/test_ear.py
python -m pytest -q experiments/xcx/tests/test_ear_reps.py
python -m pytest -q experiments/xcx/tests/test_ear_advanced.py
python -m torch.distributed.run --standalone --nproc-per-node 2 -m pytest -q \
    experiments/xcx/tests/test_kl_aux_distributed.py
python -m torch.distributed.run --standalone --nproc-per-node 2 -m pytest -q \
    experiments/xcx/tests/test_ear_distributed.py
python -m torch.distributed.run --standalone --nproc-per-node 2 -m pytest -q \
    experiments/xcx/tests/test_ear_advanced_distributed.py
```
