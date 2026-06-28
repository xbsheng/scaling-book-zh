---
layout: distill
title: "How to Profile TPU Programs"
# permalink: /main/
description: "So far this series has been entirely theoretical: back-of-the-envelope calculations based on hardware rooflines. That understanding gets you far but a lot of optimization comes down to practical details: how the XLA compiler works and how to use profiling tools like the JAX/TensorBoard Profiler to figure out what to do when it fails. We discuss this here."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference"
previous_section_name: "Part 8: Serving LLaMA"

next_section_url: ../jax-stuff
next_section_name: "Part 10: JAX"

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
  - name: "A Thousand-Foot View of the TPU Software Stack"
  - name: "The JAX Profiler: A Multi-Purpose TPU Profiler"
  - subsections:
    - name: "Trace Viewer"
    - name: "How to read an XLA op"
    - name: "Graph Viewer"
    - name: "Looking at a real(ish) example profile"
    - name: "Memory Profile"
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
---

## A Thousand-Foot View of the TPU Software Stack
Google提供了一系列用于TPU编程的API，涵盖从高级JAX代码到低级Pallas或HLO。大多数开发者会专门使用JAX代码编写程序，这样就能以抽象的NumPy风格编写线性代数程序，并自动编译为可在TPU上高效运行的形式。

以下是一个简单示例——通过JAX程序实现两个矩阵相乘：

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

通过调用`jax.jit`，我们指示JAX追踪该函数并生成名为[StableHLO](https://openxla.org/stablehlo)的低级IR，这是一种平台无关的机器学习计算表示形式，随后由XLA编译器降级为HLO。编译器运行多轮处理以确定融合策略、数据布局等因素，最终生成在JAX性能分析中可观测的HLO。这种HLO以LLVM风格的图结构表示JAX代码中的所有核心线性代数操作（矩阵乘法、逐元素运算、卷积等）。例如，上述程序的HLO精简版如下<d-footnote>要获取此HLO，可执行`jax.jit(f).lower(*args, **kwargs).compile().as_text()`</d-footnote>：

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

我们稍后会解释HLO语法，但请注意它实际上与上方的JAX代码高度对应。例如：

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

正是上述在第0和第1维度分别对两个f32矩阵执行的矩阵乘法。

**为将此HLO转换为可在TPU执行的代码，XLA编译器首先将其降级为LLO**（低级优化器）IR。LLO直接对TPU进行编程，调度存储器间数据拷贝、将数组推送至脉动阵列等操作。LLO代码包含将缓冲区推入脉动阵列、提取结果、调度不同TPU存储器间通信的DMA等原语。完成LLO降级后，再编译为加载至TPU指令存储器并执行的机器码。

当程序运行效率未达预期时，我们主要通过JAX层优化性能。但这个过程往往需要理解部分HLO语义及代码在TPU上的实际运行机制。当底层出现问题时，我们则启动另一条应急通道，使用[Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)编写自定义内核。通过JAX性能分析器可查看程序的HLO及其运行时统计信息。
## The JAX Profiler: A Multi-Purpose TPU Profiler
JAX 提供了一个多功能 TPU 分析器（profiler），其中包含一系列实用工具，可帮助用户理解程序运行时在 TPU 上发生的情况。您可以使用 `jax.profiler` 模块来跟踪程序的运行过程，并记录从每个子组件的耗时、每个程序的 HLO、内存使用情况等所有信息。例如，以下代码会将跟踪数据导出到 `/tmp/tensorboard` 目录下的文件中，该文件可在 TensorBoard 中查看（[此处](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling)提供逐步指南）。

```py
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# 现在您可以在 Google Colab 中加载 TensorBoard：
#
# !pip install -U xprof
# !pip install -U protobuf
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# 或在外部环境中运行：
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

以下是您可以在分析器中执行的操作概述：

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

进入 TensorBoard 后，分析器有几个关键选项卡，可帮助您理解程序：

1.  **跟踪查看器（Trace Viewer）** 显示 TPU 上实际发生事件的详细时间线。
2.  **图查看器（Graph Viewer）** 显示 HLO（高级优化器）图，让您了解程序各部分如何相互关联以及数据如何分片。
3.  **内存分析与内存查看器（Memory Profile and Memory Viewer）**：这些显示程序使用的内存量。

虽然分享分析数据有些困难，但[此处](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)提供了一个 Perfetto 链接，其中至少包含一个简单 Transformer 的跟踪查看器组件。[此 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 可让您生成完整的 JAX/TensorBoard 跟踪数据并进行交互。

### 跟踪查看器

**跟踪查看器可能是分析器中最有用的部分。** 下面的示例展示了一个带有标注部分的简单 Transformer。名称来源于代码中提供的标签。

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

跟踪查看器按时间顺序显示每个 TPU 核心上的所有操作。这里我们只查看 TPU:0，因为通常所有 TPU 执行相同的指令。几个关键注意事项：

1.  顶行（XLA Ops）显示实际的 TPU 操作（名称是 HLO 名称）。其他所有内容都是基于 `jax.named_scope`、`jax.named_call` 和 Python 堆栈跟踪的近似跟踪。
2.  通过注意到重复的块，我们可以在此处隔离出单个层。我们还可以通过（查看代码/理解 Transformer 的工作原理）看出哪些部分是注意力（attention）机制，哪些部分是多层感知机（MLP）。
3.  点击一个 XLA 操作，我们可以查看它在代码中的来源（有助于理解跟踪数据），并看到指向图查看器的链接。

<p markdown=1 class="takeaway">**提示：** 您可以使用“视频游戏”式控件来导航跟踪查看器，使用 A/D 左右平移，使用 W/S 缩放。这些控件使导航变得更加容易。</p>

### 如何解读 XLA 操作

HLO 其实并不难读，而且它对于理解上述跟踪中某个特定部分对应什么非常有帮助。这里有一个名为 fusion.3 的操作示例。

```c
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

让我们将其分解为几个部分。

*   **操作名称（Op Name）**: fusion.3
    *   一个点积或融合运算（fusion op）是一组操作，最多包含一次矩阵乘法，以及可能的一系列相关的逐元素 VPU 操作。
*   **形状（Shape）**: `bf16[32,32,4096]`
    *   这是操作的输出形状。我们可以看到数据类型是 bf16（每个元素 2 字节），`[32,32,4096]` 是形状。
*   **内存布局（Layout）:** `{2,1,0:T(8,128)(2,1)}`
    *   `{2,1,0:T(8,128)(2,1)}` 告诉我们轴在内存中的顺序（列主序、行主序等）以及数组填充（padding）方式。更多内容见下文。
*   **内存位置（Memory location）**: S(1)
    *   S(1) 告诉我们此数组位于 VMEM（向量内存）中。S(0)（有时省略）表示 HBM（高带宽内存）。S(2) 和 S(3) 是其他内存空间。
*   **参数（Arguments）**: `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
    *   此操作有一个输入，是一个名为 fusion.32 的 bf16 数组，具有特定形状。这告诉我们哪个函数输出是当前函数的输入。

让我们尝试更深入地理解这种表示法。以这个简单例子为例：

`f32[3,5]{1,0:T(2,2)}`

它同样告诉我们此操作返回一个形状为 `[3, 5]` 的 float32 数组，具有特定的分块（tiling）方式 `{1,0:T(2,2)}`。虽然分块方式不**那么**重要，但简而言之，分块告诉我们一个 N 维数组在内存中是如何顺序排列的。下图展示了此数组的布局：

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

在 `{1,0:T(2,2)}` 中，`1,0` 部分告诉我们数组维度在物理内存中的顺序，从最低有效位到最高有效位。您可以从右向左读取这部分，并对应 `f32[3,5]` 中的维度，以确定数组的物理布局。在此示例中，物理布局是 `[3,5]`，与逻辑形状相同。
之后，`T(2,2)` 告诉我们数组被分块为 `(2, 2)` 的块，其中每个块内，数组先行存储（**行主序**），然后列存储，即 `(0, 0)` 之后是 `(0, 1)`，然后是 `(1, 0)` 和 `(1, 1)`。由于 `T(2, 2)` 分块，数组被填充为 `[4, 6]`，内存使用量增加了约 1.6 倍。对于上面给出的大型 bf16 数组 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`，我们使用 `T(8,128)(2,1)`，这告诉我们数组有两层分块，外层的 `(8, 128)` 分块和其内的内层 `(2, 1)` 分块（用于 bf16，因此我们的加载量总是 4 字节的倍数）。例如，这里是 `bf16[4,8]{1,0:T(2,4)(2,1)}`（颜色代表 (2,4) 块，红框代表 (2,1) 块）：

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

分块会影响将张量块加载到 VMEM 的效率，有时 XLA 会在程序中引入副本，对张量进行“重新分块”或“重新布局”，有时会产生显著的开销。<d-footnote>JAX 提供了一个<a href="https://docs.jax.dev/en/latest/notebooks/layout.html">实验性功能</a>来解决这个问题，它允许 XLA 计算程序输入的“首选”布局。当您使用 `jax.jit` 对程序进行“即时”编译时，通常会传入“模拟”输入来告知 JAX 期望的形状和数据类型。这些输入通常也携带了可能并非最优的分块信息。作为替代，您可以将输入布局指定为 AUTO，`jax.jit` 将返回经过即时编译的程序所偏好的布局。然后您可以显式地以该布局加载张量，以避免在程序中引发副本。</d-footnote>

### 图查看器

虽然上面的一些融合运算看起来很复杂，但 XLA 图查看器使它们更容易解析。例如，这是一个相当复杂的融合运算的视图：

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

仔细研究大量的 HLO 图并尝试将 HLO 操作映射到您正在分析的代码中非常有帮助。将鼠标悬停在一个框上，通常会看到定义该函数的代码行。

### 查看一个（接近）真实的分析示例

[此 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 包含一个用于模拟 Transformer 的示例分析数据。如果您赶时间，[这里](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)是一个至少可以查看跟踪查看器的 Perfetto 链接。我比平时更费心地用 `jax.named_scope` 调用标注了跟踪数据，以便您识别正在发生的事情。

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

查看分析数据并尝试真正理解每个部分在做什么。让我们稍微分解一下，从 FFW（前馈网络）块开始：

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

这里我们放大了 FFW 块。您会看到上投影操作是一个融合运算（矩阵乘法），输入为 `bf16[8, 1024, 8192]` 和 `bf16[8192, 16384]`，输出为 `bf16[8, 1024, 16384]`。我知道（因为这段代码是我写的）这是一个 4 路数据并行（DP）、2 路模型并行（MP）分片矩阵乘法的局部视图，所以我们实际上在做的是：

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**我们预计这需要多长时间？** 首先，每个数据并行分片的批大小是 `8 * 1024 = 8192`，所以我们应该完全受限于计算。这是在 8 个 TPU v2 核心上运行（在 Google Colab 上免费提供），所以我们预计大约需要 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6 毫秒`，这几乎正好是实际耗时（96 毫秒）。太棒了！这意味着我们获得了极佳的浮点运算利用率！

**通信方面呢？** 您会注意到第二个矩阵乘法末尾隐藏着一个小的融合运算。如果我们点击它，您会看到

```c
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

这基本上是一个小的 ReduceScatter（这是图查看器中的视图）：

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

我们预计这需要多长时间？嗯，我们在一个 4x2 的 TPU v2 上执行 ReduceScatter，这应该只需要在 1.2e11 双向带宽上进行一次跳转。数组大小为 `2*32*1024*8192`，批次轴沿 4 路分片，因此每个分片为 `2*8*1024*8192=128MB`。所以这应该大约需要 1.1 毫秒。**实际需要多长时间？** 分析报告中显示为 1.13 毫秒。所以我们非常接近理论峰值性能！

**让我们也看看注意力机制！** 这是注意力组件的分析视图：

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

我点击了 Q 投影操作，它使用了一个形状为 [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256] 的矩阵 $$W_Q$$。我们沿着注意力头维度使用 Megatron 分片方式。尝试进行同样的计算，看看这些操作应该需要多长时间。

### 内存分析

内存分析（Memory Profile）可以轻松查看程序内存随时间的变化情况。这对于调试内存溢出（OOM）问题很有帮助。您可以看到这里大约有 7.5GB 分配给了模型参数，还有约 8.5GB 空闲。因此我们还可以在内存中放入更多内容。

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}
## Worked Problems
**问题1**：查看[这个](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/性能分析，找出可疑之处并分析其工作原理。请准确说明正在执行哪些计算，每个操作具体做了什么？参与计算的每个矩阵的真实形状是什么，它们如何被分片？*请先尝试仅通过分析性能分析结果来回答，暂时不要阅读代码。*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 点击此处查看答案。 %}

这是两次矩阵乘法运算，具体如下所示：

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

您可以看到一个约简操作、两个大型融合操作和一个全归约操作。第一个大型融合操作是：

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

这表明每个分片（shard）的形状为 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`（在8192维度上进行）。通过观察最终的全归约操作（AllReduce）及其副本组（{% raw %}`replica_groups={{0,16,32,48,64,80,96,112}, ...}`{% endraw %}），我们可以推断系统正在执行8路模型并行（model parallelism）。因此，真实的矩阵形状为 `bf16[8, 8192] * bf16[32768, 8192] -> bf16[8, 32768]`。

{% enddetails %}

**问题2**：[先前的Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing) 实现了一个简单的模拟Transformer。请按照Colab中的说明操作，获取采用GSPMD分区（partitioning）的朴素Transformer的基准测试结果。每个部分耗时多久？理论上应该耗时多久？使用了哪种分片策略？尝试修复分片问题！*提示：使用 `jax.lax.with_sharding_constraint` 来约束分片行为。修复后，您能获得的峰值矩阵单元（MXU）利用率是多少？*

作为参考，初始版本每层大约耗时184毫秒，优化后的性能分析显示每层耗时67毫秒。完成此操作后，请尝试仔细研究性能分析结果，看是否能仅凭分析结果回答以下问题：
- 采用的是何种分片策略？
- 批次大小（batch size）、模型维度（$$d_\text{model}$$）、前馈网络维度（$$d_\text{ff}$$）分别是多少？
- 注意力机制（attention）与多层感知机（MLP）模块分别占据了总耗时的多少比例？
- 根据屋顶线模型（roofline），每个操作理论上应占多少比例的时间？

**注意**：自编写本问题以来，XLA编译器已有所改进。初始版本现在每层耗时约为90毫秒，而优化后的性能分析显示每层仅提升约10毫秒（达到80毫秒/层）。尽管如此，这仍然值得尝试，看看您能否做得更好。

<h3 markdown=1 class="next-section">第9部分到此结束。要深入探讨JAX并行计算的第10部分，请点击[此处](../jax-stuff)。</h3>