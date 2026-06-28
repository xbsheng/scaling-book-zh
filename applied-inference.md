---
layout: distill
title: "Serving LLaMA 3-70B on TPUs"
# permalink: /main/
description: "Let's take a close look at how we'd serve LLaMA 3-70B models on TPU v5e. How expensive are different models to serve at roofline? How large are their KV caches? What batch sizes should we use? How are the parameters and activations sharded during inference? Let's work through some back-of-the-envelope estimates for latency and throughput in production."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling
next_section_name: "Part 9: Profiling"

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "What's the LLaMA Serving Story?"
  - subsections:
    - name: "Thinking about throughput"
    - name: "What about prefill?"
  - name: "Visualizing the Latency Throughput Tradeoff"
  - name: "Worked Problems"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---*本节将探讨部署LLaMA-3模型所需的条件及其高效实现的途径。如同之前的"应用"章节，建议您先尝试自行推导答案（可准备纸笔演算），再参考解析！*
## What's the LLaMA Serving Story?
让我们回顾一下 LLaMA 3-70B 的架构参数（参考[第 6 节](../applied-training)）：

| **超参数 (hyperparam)**              | **值 (value)** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

我们从一个简单的问题开始：**我们应该在什么硬件上提供服务？** 答案基本上是，选择每美元浮点运算次数 (FLOPs / dollar) 最便宜的那个。<d-footnote>这并不总是成立，有时高带宽内存 (HBM) 或ICI 带宽比 FLOPs 更关键，但这是一个很好的启发式方法。</d-footnote> 因此，我们通常希望在我们的专用推理芯片 TPU v5e 上提供服务（成本数据来自截至 2025 年 2 月的 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元/小时** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

每个 TPU v5e 配备 16GB 的 HBM，这要求我们对模型进行相当激进的分片。让我们先思考一些对我们可能很重要的基本量：

**问题：** LLaMA 3-70B 的每个 token 的 KV 缓存 (KV cache) 有多大？*你可以假设我们使用 int8 格式存储它们。这决定了我们在给定拓扑结构上能使用的批大小 (batch size)。*

<details><summary>如果你思考过了，请点击这里！</summary>


LLaMA 3-70B 有 8 个 KV 头，所以每个 token 的大小为 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**注意这有多大！** 如果我们有一个长度为 32k token 的序列（这很常见），这将使用 `160e3 * 32,768 = 5.3GB / 序列`。对于批大小 (BS) = 240，这就是 1.3TB！由于 TPU v5e 每个只有 16GB 内存，我们大约需要 `(70e9 + 1.3e12) / 16e9 = 86` 个 TPU v5e 芯片才能容纳这么大的内存。同时请注意，与 70GB 的模型参数相比，这有多大。


</details>

**问题：** 假设我们想在批大小为 32、序列长度为 8192 的情况下服务 L3 70B，并且所有内容（参数和 KV 缓存）都使用 int8 格式。这将使用多少总内存？我们能在多小的切片 (slice) 上提供服务？

<details><summary>答案</summary>


由于我们的 KV 缓存在 int8 下大小为 `160e3` 字节，我们的总 KV 内存为 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数为 `70e9` 字节，因为每个参数占 1 字节。因此，我们的总内存使用量为 `41.9e9 + 70e9 = 112GB`。

我们能使用的最小切片需要 `112e9 / 16e9 = 7` 个 TPU，或者（四舍五入到偶数大小）一个 TPU v5e `4x2`。这会很紧张，考虑到其他开销，我们可能无法完全容纳，所以我们可能至少需要一个 `4x4` 的切片（或者减小批大小）。


</details>

**问题：** 在这个批大小和量化设置下，在 TPU v5e `4x2` 切片上，我们预期的每个解码步骤 (decode step) 延迟大约是多少？吞吐量（tokens / sec / chip）呢？`4x4` 呢？*假设我们在 bfloat16 中执行浮点运算，并且所有内容都完全分片。*

<details><summary>答案</summary>


我们可以调用上一节的公式：

$$\begin{align*}
\tiny \text{理论步骤时间（通用）} = \underbrace{\frac{\text{批大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力机制（总是带宽受限）}} + \underbrace{\max\left(\frac{2 \times \text{批大小} \times \text{参数数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能是计算受限）}}
\end{align*}$$

这里我们的临界批大小 (critical batch size) 大约是 120，因为我们的参数是 int8 但我们的 FLOPs 是 bfloat16。我们也可以手动计算右侧的最大值，但这基本上是我们已经做过几次的计算。**所以我们的矩阵乘法和 FLOPs 都深入到了内存受限 (memory-bound) 的领域。**

严格看内存带宽，我们的步骤时间基本上是 `(KV 大小 + 参数大小) / (8 * HBM 带宽) = 112e9 / (8 * 8.2e11) = 17ms`。**所以理论上我们的步骤时间约为 17ms。** 我们的吞吐量将是 `32 / .017 = 1882 tokens / sec`，或 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个注意事项，需要检查我们的矩阵乘法是否可能受限于 ICI 带宽。我们这里可以分配 2 个轴给它，所以理论上当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时，我们会受限于 ICI 带宽，所以没问题！

如果我们在 `4x4` 上运行，ICI 方面仍然没问题，所以我们的延迟会降到 `17 / 2 = 8.5ms`，但每芯片的吞吐量将保持不变。


</details>

### 思考吞吐量

让我们花点时间纯粹思考一下吞吐量。当我们优化吞吐量时，我们希望成为计算受限 (compute bound)，这意味着我们接近充分利用 TPU MXU 的容量。通常这意味着我们希望批大小尽可能大，这样我们就能完成尽可能多的工作。

**问题：** 在 TPU v5e 上，使用 bfloat16 权重和激活值，我们的批大小需要多大才能使我们的矩阵乘法成为计算受限？如果我们使用 int8 权重但在 bfloat16 中执行 FLOPs 呢？如果是 int8 权重和 int8 FLOPs 呢？

<details><summary>答案</summary>


正如在第 7 节所讨论的，对于任何 $B \ll D, F$ 的 bfloat16 矩阵乘法，我们有：

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM 带宽}} = 240
\end{equation*}$$

当我们的权重是 int8 时，分母会损失一个 2 的因子，所以我们有 $2BDF / DF = 2B > 240$，或者说 $B > 120$，是之前临界批大小的一半。这对我们很有帮助！当我们使用 int8 权重和 int8 FLOPs 时，我们必须使用 TPU FLOPs/s 的 int8 值，它从 bfloat16 的 1.97e14 增加到 3.94e14，几乎翻倍。这意味着我们回到了大约 $B > 240$ 的起点。

int8 权重和 bfloat16 FLOPs 的情况很常见，因为无损量化参数通常比进行低精度算术更容易。


</details>

**问题：** 使用 bfloat16、int8 和 int4（KV 缓存和参数都是）以及 8k 上下文长度，我们能在多小的 TPU v5e 拓扑结构上服务 LLaMA 3-70B？*对于这个问题，你可以认为 KV 缓存小到可以忽略不计。*

<details><summary>答案</summary>


这很简单！如果我们能接受一个很小的批大小，那么唯一的限制就是将参数内存放入 HBM 中，即只需 `ceil(num_params * sizeof(dtype) / 每个 TPU 的 HBM)`，或 `ceil(70e9 * sizeof(dtype) / 16e9)` 四舍五入到最近的合理拓扑结构（2 的倍数）：

| dtype | 参数大小 | KV 大小 / token (字节) | 最小 TPU v5e 数量 | 实际最小切片 | 剩余 HBM 用于 KV 缓存 | 8k 下的 KV 缓存数量 |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 芯片   |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 芯片    |             58              |         43         |
| int4  |    35GB    |          81kB           |     2.81     |  2x2 = 4 芯片    |             29              |         43         |

这很酷！它告诉我们，如果我们愿意，可以将 LLaMA 70B 放在一个 TPU v5e 2x2 切片上。只是你会注意到 KV 缓存的数量非常小。这就是我们的批大小！这意味着我们的浮点运算利用率 (FLOPs utilization) 会很差。我们很乐意使用更大的拓扑结构来将批大小提高到 240。


</details>

**问题：** 假设我们使用这些拓扑结构所能容纳的最大批大小，我们预期的每个生成步骤的延迟是多少？

<details><summary>答案</summary>


这也简单，因为我们选择的批大小是为了填满所有的 HBM！这只是将一整个 TPU v5e 大小的字节加载到 MXU 需要多长时间的问题。这只是 `v5e HBM / v5e HBM 内存带宽 = 16GB / 8.2e11 = 19ms`，所以这是 **19ms / 步**。假设我们的生成序列的中位长度为 512 个 token，那么每个解码大约需要 9 秒。请注意，使用更小的批大小我们可以获得稍好的延迟，例如，如果我们只看 int4 的模型参数，我们的最小延迟大约是 10ms / 步，因为 HBM 不再是满的。


</details>

<p markdown=1 class="takeaway">**要点**：我们总是可以通过询问从 HBM 将模型所有参数加载到 MXU 需要多长时间来获得解码延迟的下界。当我们的 KV 缓存较小时，你可以认为每一层只是逐块加载权重然后丢弃它们。除非我们使用大的批大小或大量的设备间通信，这通常是一个合理的下界（误差在 1.5 倍以内）。当我们的批大小更大时，我们也需要建模 KV 缓存的加载，因为 KV 缓存会主导参数。</p>

同样，在 FLOPs 受限（compute-bound）的领域（例如训练或大批次推理），我们可以使用 $$\text{总 FLOPs} / (N \cdot C) = 2 \cdot \text{参数数量} \cdot B / (N \cdot C)$$ 作为下界，这假设没有通信开销。

**问题：** 对于每种情况，这给我们带来了每芯片多少吞吐量（以 queries / chip 为单位）？*你可以假设我们的解码序列中位长度为 512 个 token。*

<details><summary>答案</summary>


这是一个重要的问题，因为它直接关系到每 token 的成本。

根据我们对中位解码长度的假设，我们的吞吐量就是 $$B / (\text{每步延迟} \cdot \text{中位步数} \cdot N) \approx 43 / (0.019 * 512 * N)$$。这大约给我们 $$4.42 / N$$ QPS，所以代入 $$N$$ 我们得到：

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.55    |
|   int4   |    1.11    |

请注意，这相当乐观，因为它完全忽略了前向传播的工作内存（分配给激活值和注意力机制的内存）。对于 Flash Attention 来说这不算荒谬，但也不现实。真实的数字可能大约是这些值的一半。为了获得绝对最大吞吐量，我们可能需要将芯片数量增加一倍以上，并显著增加批大小。


</details>

**问题：** 如果我们对上面每个例子的拓扑结构加倍，我们的峰值吞吐量会如何变化？

<details><summary>答案</summary>


如果我们在 bfloat16 中使用 4x8 切片，我们将有 372GB 剩余用于 KV 缓存，这将使我们能够将批大小增加到 140。然后，由于我们的步骤时间将保持不变，我们的吞吐量将是 `14.39 / 芯片数`，或

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.44    |
|   int8 (on 4x4)   |    0.90    |
|   int4 (on 2x4)   |    1.80    |

进一步增加将带来更大的收益！**关键点在于，在所有情况下，最小的拓扑结构并不一定是性能最高的拓扑结构**，特别是如果我们受到 KV 缓存大小的限制。


</details>

**问题：** 现在让我们深入探讨分片 (sharding) 的问题。假设我们想在 TPU v5e 4x8 上使用 bfloat16 进行服务。在生成期间，我们将在 TPU v5e 4x8 上使用什么样的模型分片？我们能避免通信受限 (communication bound) 吗？

<details><summary>答案</summary>


正如上一节所讨论的，在生成期间我们只有一种分片选择：模型并行 (model parallelism)。在变得通信受限之前，我们能做多少？正如我们在上一节讨论的，我们的模型变得通信受限大致在

$$Y > \frac{F \cdot M_Y}{2200}$$

对于 LLaMA 3-70B，我们有 `F = 28,672`，所以如果我们做 2 个轴的模型分片，这大约给我们 $$Y = 28672 * 2 / 2200 =
## Visualizing the Latency Throughput Tradeoff
继续沿用LLaMA 70B模型，现在来实际观察生成过程中不同批处理大小下的延迟和吞吐量。如前面对PaLM模型的分析所示，这为我们提供了吞吐量/延迟的帕累托前沿（Pareto frontier）。我们假设采用16路张量并行（tensor parallelism），因为这是在MLP模块中保持计算密集型时的合理界限。此处将使用TPU v5e 4×4拓扑结构。**滑动条可控制序列长度，以便观察更大KV缓存带来的影响。**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **请注意成本与延迟之间的权衡多么显著。** 以每token延迟翻倍为代价，我们可实现约100倍的每token成本降低。此外，延迟范围可从低批处理大小时的5.5ms到极大批次时的20ms。
* 注意当序列长度为2k时，吞吐量在达到BS 120上限后趋于稳定（此处为120因采用int8权重但bf16浮点运算）。然而随着序列长度增加，内存无法容纳此批次大小，因此永远不会达到完全饱和点。
* 请注意在相同吞吐量下，大批次处理的延迟明显更高，因为KV加载（而非参数加载）成为主导因素。

通过将成本和延迟来源分解为参数加载时间、KV加载时间和浮点运算时间，我们能更清晰地理解这一点。红色区域表示预计在MLP模块中受计算限制的部分。

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

这揭示了重要规律：初始阶段参数加载构成延迟主体，直到批处理大小足够大，浮点运算和KV加载才变得显著。特别值得注意的是，**当序列长度超过2048时，KV缓存加载耗时始终超过浮点运算耗时！因此虽然增大批次可提升硬件利用率，但在长上下文场景中KV加载始终主导总步骤时间。**

<p markdown=1 class="takeaway">**核心结论：** 对于LLaMA 3-70B模型，在几乎所有配置中我们都受到KV缓存内存带宽（及HBM）的强烈限制，这凸显了缩减KV缓存尺寸对生成吞吐量的重要性。同时需注意此处延迟/吞吐量的权衡效应依然非常显著。</p>

<details><summary>相关代码实现如下</summary>


以下是计算性能上限的代码：

```py
import numpy as np

num_chips = 16  # 固定总模型并行度为16
bytes_per_param = 1  # int8表示每参数1字节
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192  # 可调整此参数

hbm_bandwidth = 8.20E+11  # v5e带宽
flops = 1.97E+14  # v5e算力

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(
    num_chips: int,
    sequence_length: int,
    param_size: float,
) -> int:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  required_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(required_chips <= num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(
    num_chips=num_chips,
    sequence_length=sequence_length,
    param_size=param_size,
)  # 获取可容纳的最大批次大小
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_size * batch_sizes / (num_chips * flops)  # 二次归约场景的近似值

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # 生成阶段始终受带宽限制

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

请注意代码如何明确将延迟分解为两个来源：KV加载和参数加载，且延迟取决于浮点运算或通信中耗时更长的一方。


</details>
## Worked Problems
以下是几个已解答的问题。部分内容可能与前文有所重复，但在教学上可能有所助益。

**问题 1：** LLaMA 3-405B 的每个前向传播每词元使用多少次浮点运算（FLOPs）？假设我们处于浮点运算受限状态，在 TPU v5e 的 N 个芯片上单次前向传播的下限是多少？如果是通信受限状态呢？*请忽略模型无法在单个芯片上运行的情况。*

**问题 2：** 假设我们希望使用 int8 权重和 int8 KV 缓存，以 BS240（批量大小 240）来服务 LLaMA 3-8B。以下各项各占用多少字节：(a) 模型参数 (b) KV 缓存 (c) 峰值工作激活值（大致）？我们能在其上运行此服务的最小拓扑是什么？

**问题 3：** 你如何在 TPU v5e 上服务 LLaMA 3-405B？假设使用 int8 权重和 bfloat16 浮点运算。如果我们对每词元有严格的 15 毫秒限制，我们能达到的最高吞吐量配置是什么？理论上的最小步长时间是多少？

<h3 markdown=1 class="next-section">第 8 部分到此结束！关于第 9 部分，深入探讨 XLA 和 TPU 性能分析，请点击 [此处](../profiling)。</h3>