---
layout: distill
title: "Sharded Matrices and How to Multiply Them"
# permalink: /main/
description: "When we train large ML models, we have to split (or \"shard\") their parameters or inputs across many accelerators. Since LLMs are mostly made up of matrix multiplications, understanding this boils down to understanding how to multiply matrices when they're split across devices. We develop a simple theory of sharded matrix multiplication based on the cost of TPU communication primitives."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus"
previous_section_name: "Part 2: TPUs"

next_section_url: ../transformers
next_section_name: "Part 4: Transformer Math"

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
  - name: "Partitioning Notation and Collective Operations"
  - subsections:
    - name: "A unified notation for sharding"
    - name: "How do we describe this in code?"
  - name: "Computation With Sharded Arrays"
  - subsections:
    - name: "Case 1: neither multiplicand has a sharded contracting dimension"
    - name: "Case 2: one multiplicand has a sharded contracting dimension"
    - name: "Case 3: both multiplicands have sharded contracting dimensions"
    - name: "Case 4: both multiplicands have a non-contracting dimension sharded along the same axis"
  - name: "A Deeper Dive into TPU Communication Primitives"
  - subsections:
    - name: "Our final communication primitive: the AllToAll"
    - name: "More about the ReduceScatter"
    - name: "How to overlap matmul communication with compute"
  - name: "What Have We Learned?"
  - name: "Some Problems to Work"

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

## Partitioning Notation and Collective Operations
当我们使用成千上万个 TPU 或 GPU 训练大语言模型时，从抽象层面讲，执行的计算与在单设备上训练时是相同的。区别在于 **我们的数组无法放入单个 TPU/GPU 的高带宽内存中**，因此必须对其进行分割。<d-footnote>值得注意的是，我们有时也会选择为了提升速度而进行并行化。即使模型能够放入更少的芯片上，扩展到更多芯片也能带来更高的每秒浮点运算次数。例如在推理阶段，有时可以使用更小的拓扑结构，但为了降低延迟，我们仍会选择扩展到更大的拓扑结构。同样在训练中，为了缩短每步时间，我们也常将模型扩展到更多芯片上。</d-footnote> 我们将这种操作称为数组的 *分片* 或 *分区*。模型扩展的艺术在于，如何设计分片方案以确保计算保持高效。

下面是一个二维数组 **A** 跨 4 个 TPU 进行分片的示例：

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>图示：</b>一个形状为 <b>A</b>[I, J] 的示例数组在 4 个设备上进行分片。两个维度均通过分片方案 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>] 在 2 个设备上均匀分割。每个 TPU 持有总内存的 1/4。" %}

请注意，分片后的数组与未分片数组具有相同的 *全局* 或 *逻辑形状*，例如 `(4, 128)`，但它还有一个 *设备本地形状*，如 `(2, 64)`，这表示每个 TPU 实际持有的字节大小（上图中每个 TPU 持有总数组的 ¼）。现在，我们将此推广到任意数组。

### 统一的分片符号表示

我们使用一种 *命名轴符号* 的变体来描述张量 *如何* 以块的形式分布在各设备上：我们假设存在一个二维或三维的设备网格，称为 **设备网格**，其中每个轴都被赋予了 **网格轴名称**，**例如 X、Y 和 Z**。通过描述数组每个命名维度在物理网格轴上的划分方式，我们就能指定矩阵数据在设备网格上的布局。我们将这种分配方案称为 **分片**。

**示例（上图）**：对于上图，我们有：
* **网格**：上述设备网格 `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`，这表示我们拥有一个 2x2 的网格，包含 4 个 TPU，轴名称为 $X$ 和 $Y$。
* **分片**：$A[I_X, J_Y]$，这表示将第一个轴 $I$ 沿网格轴 $X$ 进行分片，将第二个轴 $J$ 沿网格轴 $Y$ 进行分片。这个分片方案告诉我们每个分片持有数组的 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$。

综合起来，我们知道数组的本地形状（单个设备持有的分片大小）是 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$，其中 $$\lvert I\rvert$$ 是 A 第一个维度的大小，$$\lvert J\rvert$$ 是 A 第二个维度的大小。

<b markdown=1 style="color: #048affff;">小测验 [跨 1 个轴的二维分片]：</b> 考虑一个数组 `fp32[1024, 4096]`，分片方案为 $A[I_{XY}, J]$，网格为 `{'X': 8, 'Y': 2}`。每个设备持有多少数据？在 H100 上从高带宽内存加载此数组需要多长时间（假设单芯片内存带宽为 `3.4e12`）？

{% details 点击这里查看答案。 %}

$A[I_{XY}, J]$ 将第一个维度 (I) 同时沿 X 和 Y 硬件轴进行分片。在本例中，本地形状为 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$。对于给定示例，全局形状为 `fp32[1024, 4096]`，因此本地形状为 `fp32[64, 4096]`。

由于每个 GPU 持有 `4 * 64 * 4096 = 1MiB` 字节，加载时间大约为 `1e6 / 3.4e12 = 294ns`，不过由于数据量很小，加上各种开销，实际时间可能会显著更长。

{% enddetails %}

**可视化这些分片**：让我们尝试通过观察一个分布在 4 个设备上的二维数据数组来可视化这些分片：

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

矩阵的 *完全复制* 形式我们简单地写作 $A[I, J]$，不附加任何分片分配。这意味着 *每个* 设备都包含整个矩阵的完整副本。

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

我们可以用一个下标网格轴来指示其中一个维度已在某个网格轴上被划分。例如，$A[I_X, J]$ 表示 **I** 逻辑轴已在 **X** 网格维度上被划分，但 **J** 维度 *未被* 划分，并且数据块在 **Y** 网格轴上保持 *部分复制*。

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$ 表示 **I** 逻辑轴已在 **X** 网格轴上被划分，**J** 维度已在 **Y** 网格轴上被划分。

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

我们在下图中说明了其他可能性：

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

这里 $A[I_{XY}, J]$ 表示我们将 **X** 和 **Y** 网格轴视为一个更大的扁平化维度，并将 **I** 命名轴在所有设备上进行划分。多个网格轴下标的顺序很重要，因为它指定了跨网格划分的遍历顺序。

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

最后，请注意我们 *不能* 让多个命名轴沿 *同一个* 网格维度进行分片。例如，$A[I_X, J_X]$ 是一个无意义且禁止的分片方案。一旦一个网格维度被用于划分数组的某个维度，它在某种意义上就被“消耗”了。

<b markdown=1 style="color: #57cf57;">小测验：</b> 设 **A** 为形状为 `int8[128, 2048]` 的数组，分片方案为 $A[I_{XY}, J]$，网格为 `Mesh({'X': 2, 'Y': 8, 'Z': 2})`（共 32 个设备）。**A** 在每个设备上使用多少内存？**A** 在所有设备上总共使用多少内存？

{% details 点击这里查看答案。 %}

**答案**：我们的数组 **A** 在 X 和 Y 上进行分片，在 Z 上进行复制。因此，每个设备上的形状为 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`，大小为 `8 * 2048 = 16,384` 字节。因为它在 Z 上进行复制，同时在每个 Z 平面内又在 X 和 Y 上完全分片，所以原始数组有 2 个完整的副本（每个 Z 平面一个）。因此，所有设备上的总大小为：原始数组大小 × Z 副本数 = 128 * 2048 * 2 = 512 KiB。或者，我们可以验证：32 个设备 × 每设备 16,384 字节 = 总共 512 KiB。

{% enddetails %}

### 如何在代码中描述？

到目前为止，我们一直避免谈论代码，但现在是一个抢先了解的好机会。JAX 使用一种命名分片语法，它与我们上面描述的抽象语法非常匹配。我们将在[第 10 节](../jax-stuff)中详细讨论，这里先快速预览一下。你可以在一个 [Google Colab](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing) 中尝试，并对结果进行性能分析，以了解 JAX 如何处理不同的分片。这段代码片段做了 3 件事：

1.  创建一个 **jax.Mesh**，将我们的 8 个 TPU 映射到一个 4x2 的网格中，并为两个轴分配名称 'X' 和 'Y'。
2.  创建矩阵 A 和 B，其中 A 沿其两个维度分片，B 沿输出维度分片。
3.  编译并执行一个简单的矩阵乘法，返回一个分片数组。

```py
import jax
import jax.numpy as jnp

# 创建我们的网格！我们运行在一个 4x2 的 TPU v2-8 分片上，轴名为 'X' 和 'Y'。
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 一个辅助函数，用于定义我们的分片。PartitionSpec 是我们的
# 分片方案（一个从轴到名称的映射）。
def P(*args):
  return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# 我们将 A 和 B 都在非收缩维度上分片，并在收缩维度上对 A 分片。
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 我们可以对这些分片数组执行矩阵乘法！out_shardings 告诉我们
# 希望输出如何分片。JAX/XLA 会处理其余的分片逻辑。
y = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y'))(A, B)
```

JAX 的一个很酷的特点是，这些数组的行为就像它们没有被分片一样！`B.shape` 会告诉我们全局或逻辑形状 (2048, 8192)。我们实际上需要查看 `B.addressable_shards` 才能了解它的本地分片情况。我们可以对这些数组执行操作，JAX 将尝试找出如何广播或重塑它们以执行操作。例如，在上面的例子中，**A** 的本地形状是 `[2, 1024]`，**B** 的本地形状是 `[2048, 4096]`。JAX/XLA 会自动在这些数组之间添加必要的通信，以执行最终的乘法运算。
## Computation With Sharded Arrays
假设您有一个分布式存储在多个设备上的数据数组，并希望对其执行数学运算，那么对数据和计算进行分片（sharding）会带来哪些开销？

显然，这取决于所涉及的计算类型。

*   对于**逐元素操作**（elementwise operations），对分布式数组执行操作**没有额外开销**。
*   当我们希望对分布在多个设备上的元素执行跨元素操作时，情况就变得复杂了。幸运的是，对于大多数机器学习任务，几乎所有的计算都以矩阵乘法（matrix multiplications）的形式进行，而这相对容易分析。

本节的剩余部分将讨论如何对分片矩阵进行乘法运算。初步来看，这涉及在设备间移动矩阵的块（chunks），以便对每个块进行完整的乘法或求和操作。**每种分片方式都会涉及不同的通信。** 例如，$A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$ 可以在没有任何通信的情况下完成乘法，因为*收缩维度*（contracting dimension，即我们实际进行求和的维度 J）没有被分片。然而，如果我们希望输出是未分片的（即 $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$），那么我们需要将 $A$ 和 $B$ 或 $C$ 复制到每个设备（使用一种*全收集*（AllGather）操作）。这两种选择具有不同的通信成本，因此我们需要计算这种成本并选择最低的那个。

{% details 您可以将其视为"分块矩阵乘法"（block matrix multiplication）来理解。 %}

要理解这一点，回顾"分块矩阵"（block matrix）的概念会很有帮助，即一个矩阵的嵌套矩阵：

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

矩阵乘法有一个很好的性质：当矩阵乘数用分块表示时，其乘积可以按照标准规则用分块矩阵乘法来表示：

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

这意味着实现分布式矩阵乘法可以归结为：通过网络移动这些分片的块，对块执行*本地*矩阵乘法，然后对结果求和。**问题在于需要添加什么通信，以及其开销有多大。**

{% enddetails %}

方便的是，我们可以将所有可能的分片情况大致归纳为 4 种需要考虑的情况，每种情况都对应一条关于需要添加何种通信的规则：
1.  **[情况 1](#case-1-neither-multiplicand-has-a-sharded-contracting-dimension):** 两个输入都没有沿收缩维度分片。_我们可以对本地分片进行乘法，无需任何通信。_
2.  **[情况 2](#case-2-one-multiplicand-has-a-sharded-contracting-dimension):** 一个输入沿收缩维度分片。_我们通常沿收缩维度对分片的输入执行"全收集"（AllGather）。_
3.  **[情况 3](#case-3-both-multiplicands-have-sharded-contracting-dimensions):** 两个输入都沿收缩维度分片。_我们可以先对本地分片进行乘法，然后对结果执行"全归约"（AllReduce）。_
4.  **[情况 4](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis):** 两个输入都有一个非收缩维度沿相同的轴分片。在不先对其中一个输入执行全收集的情况下，我们无法继续操作。

您可以将这些视为需要遵循的规则，但理解这些规则成立的原因及其开销大小也很有价值。我们现在将详细讨论每一种情况。

### 情况 1：两个乘数都没有分片的收缩维度

**引理：** 当对分片矩阵进行乘法时，计算是有效的，并且输出遵循输入的分片方式，*除非*收缩维度被分片或两个矩阵沿同一轴分片。例如，下面的运算完全没问题

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

完全不需要任何通信，并且结果是一个在 X 和 Y 两个硬件维度上都分片的张量。试着思考一下为什么。基本上，计算*独立于*分片方式，因为每个批处理条目（batch entry）都有一些本地的收缩轴块可以相乘和归约。以下任何一种情况都遵循此规则并能正常工作：

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

因为 **A** 和 **B** 都没有分片的收缩维度 **J**，我们可以简单地对输入执行本地分块矩阵乘法，其结果*已经*按照期望的输出分片方式分片了。当两个乘数都有非收缩维度沿相同轴分片时，情况就不再如此了（有关详细信息，请参见[无效分片](#case-4-both-multiplicands-have-a-non-contracting-dimension-sharded-along-the-same-axis)部分）。

### 情况 2：一个乘数有分片的收缩维度

让我们考虑当一个输入 **A** 沿收缩的 **J** 维度分片，而 **B** 完全复制时该怎么做：

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

我们不能简单地将 **A** 和 **B** 的本地块相乘，因为我们需要对 **A** 的完整收缩维度求和，而这个维度被拆分到了 X 轴上。通常，我们首先对 **A** 的分片执行"**全收集**"（AllGather），这样每个设备都有一个完整的副本，然后再与 **B** 相乘：

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

这样，实际的乘法就可以在每个设备上完整地进行。

<p markdown=1 class="takeaway">**要点：** 当乘法矩阵中的一个矩阵沿收缩维度分片时，我们通常先对其进行全收集，使收缩不再被分片，然后执行本地矩阵乘法。</p>

请注意，当 **B** 没有同时沿 X 轴分片时，我们也可以先执行本地部分矩阵乘法，然后对分片的部分和进行求和（或*全归约*（AllReduce）），这样我们可以分片计算，但通常通信成本更高。在某些情况下这可能更快，尽管在实践中通常 **B** 是分片的。[下面](#some-problems-to-work)的问题 4 讨论了什么时候这种方式更好。

**什么是全收集（AllGather）？** 全收集是我们将要讨论的第一个核心 [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 通信原语。全收集*移除*沿某轴的分片，并将分布在设备上的分片重新组装到沿该轴的*每个*设备上。使用上面的符号，全收集移除一组轴上的下标，例如

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

我们不必移除给定维度的所有下标，例如 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$ 也是一个全收集，只是只在一个轴上进行。另请注意，我们可能也想使用全收集来移除*非收缩*维度的分片，例如在矩阵乘法中：

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

我们可以在开始时对 **A** 执行全收集以移除输入分片，或者我们可以执行分片的矩阵乘法，然后对结果 **C** 执行全收集。

**全收集具体是如何执行的？** 为了在一个 TPU 轴（一个环）上执行一维全收集，我们基本上是让每个 TPU 将其分片在一个环上传递，直到每个设备都有一个副本。<d-footnote>GPU 的全收集也可以这样工作，您在节点内的 GPU 之间创建一个环，并按照（任意）顺序传递数据块。</d-footnote> 这是一个动画：

{% include figure.liquid path="assets/img/all-gather.gif" caption="<b>图：</b>在一组 8 个 TPU 或 GPU 设备上执行全收集的动画。每个设备开始时持有数组的 1/8，结束时拥有完整的副本。" %}

我们可以向一个方向或两个方向执行全收集（上图显示了两个方向）。如果向一个方向，每个 TPU 发送大小为 $\text{bytes} / N$ 的块，经过环上的 $N - 1$ 跳。如果向两个方向，我们有 $\lfloor \frac{N}{2} \rfloor$ 跳，每跳大小为 $2 \cdot \text{bytes} / N$。

**这需要多长时间？** 让我们以双向全收集为例来计算所需时间。设 $$V$$ 为数组的字节数，$X$ 为收缩维度上的分片数。那么根据上图，每一跳在每个方向发送 $V / \lvert X\rvert$ 字节，因此每一跳耗时

$$T_{hop} = \frac{2 \cdot V}{\lvert X \rvert \cdot W_\text{ici}}$$

其中 $W_\text{ici}$ 是**双向** ICI 带宽。<d-footnote>分子中的因子 2 来自于我们使用的是双向带宽这一事实。我们在每个方向发送 $V / X$，总共是 $2V / X$。</d-footnote> 我们需要发送总共 $\lvert X\rvert / 2$ 跳才能到达每个 TPU<d-footnote>技术上是 $\lfloor X / 2 \rfloor$</d-footnote>，因此整个归约操作耗时

$$T_{total} = \frac{2 \cdot V \cdot X}{2 \cdot X \cdot W_\text{ici}}$$

$$T_{total} = \frac{V}{W_\text{ici}}$$

请注意，这**不依赖于 $X$**！这相当惊人，因为这意味着即使我们的 TPU 只是局部连接，连接的局部性也无关紧要。我们只是受限于每个链路的速度。

<p markdown=1 class="takeaway">**要点：** 在吞吐量受限的情况下执行全收集（或全归约（ReduceScatter）或全归约（AllReduce））时，实际的通信时间仅取决于数组的大小和可用带宽，而与我们的数组分片在多少个设备上无关！</p>

**关于 ICI 延迟的说明：** 每次跳过 ICI 链路都会有一些固有的开销，与数据量无关。这通常在 1us 左右。这意味着当我们的数组 $$A$$ 非常小，每跳耗时少于 1us 时，我们可能会进入"延迟受限"模式，此时计算时间_确实_依赖于 $X$。

{% details 欲知完整细节，请点击此处。 %}

设 $$T_\text{min}$$ 为单跳的最小时间。那么

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{X \cdot W_\text{ici}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot X}{2}, \frac{V}{W_\text{ici}} \right]$$

因为我们执行 $X / 2$ 跳。对于大的归约或收集操作，我们完全受限于带宽。我们发送如此多的数据，以至于每跳的开销实际上可以忽略不计。但对于小数组（例如，从模型中采样时），这并非可以忽略不计，ICI 带宽也无关紧要。我们纯粹受延迟限制。换句话说，对于一个特定的 TPU，例如具有 `4.5e10` 单向 ICI 带宽的 TPU v5e，发送任何小于 `4.5e10 * 1e-6 = 45kB` 的缓冲区都将是延迟受限的。

{% enddetails %}

这是在 TPU v5e 8x16 切片上对全收集带宽的实测结果。该数组在 16 轴上分片，因此它有一个完整的双向环。

{% include figure.liquid path="assets/img/all-gather-bandwidth.png" class="img-small" caption="<b>图：</b>TPU v5e 在执行全收集期间的实测带宽和估计链路带宽。橙色的 BW 是全收集的实际每秒字节数，而蓝色曲线显示了根据已知集合操作成本计算出的经验单向链路带宽。" %}

请注意，我们不仅达到了约 95% 的峰值标称带宽（`4.5e10`），而且是在约 10MB 的数据量时达到峰值，当进行 16 路分片时，每个设备约 625kB（*注：这远优于 GPU）。

**当我们在多个轴上执行全收集时会发生什么？** 当我们在多个轴上收集时，我们有多个 ICI 维度来执行收集。例如，AllGather<sub>XY</sub>([B, D<sub>XY</sub>]) 操作在两个硬件网格轴上进行。这使可用带宽增加了 $N_\text{axes}$ 倍。

考虑到延迟，我们得到一般规则：

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ici} \cdot N_\text{axes}} \right]$$

其中 $$\sum_i \lvert X_i \rvert / 2$$ 是 TPU 网格中最长路径的长度。

<b markdown=1 style
## A Deeper Dive into TPU Communication Primitives
前文四个案例已介绍了执行分片矩阵乘法所用的几种“核心通信原语”：

1. **AllGather：** 从分片中移除一个下标，收集所有分片。
2. **ReduceScatter：** 通过对特定轴上的分片求和，移除数组中一个“未归约”的后缀维度，使数组在另一轴上保持分片状态。
3. **AllReduce：** 移除一个“未归约”的后缀维度，使数组在该轴上不再分片。

还有一种在混合专家（MoE）模型及其他计算中常见的核心通信原语需要介绍：**AllToAll**。

### 最终通信原语：AllToAll

最后一个基本的集合通信原语——在考虑分片矩阵乘法时不会自然出现，但在实践中频繁使用的——是 **AllToAll** 集合通信，更精确地说，是一种特殊的*分片转置*或重分片操作。例如：

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll 通常用于在计算的不同区域之间重新排列分片布局，特别是当这些区域的布局方案不兼容时。它在考虑分片混合专家模型时会自然产生。*你可以将 AllToAll 理解为将一个下标从一个轴移动到另一个轴*。由于 AllToAll 不需要将每个分片的所有数据复制到环的所有节点上，它的开销实际上比 AllGather 更低（低至四分之一）<d-footnote>对于偶数大小的双向环，每个设备将向右发送 $(N/2 + (N/2-1) + … + 1)$ 个数据块，向左发送 $((N/2-1) + … + 1)$ 个数据块，总计 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$。每个数据块（即分片的分片）的大小为 $\text{bytes} / N^2$，因此单设备开销为 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$。该结果适用于所有设备，因为总带宽随设备数量线性扩展。</d-footnote>。

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

若推广至 N 维 AllToAll，在一个 AxBxC 的网格上，对总字节数为 $V$ 的数组（所有设备总和）的通信开销为

$$T_\text{comms per AllToAll} = \frac{V \cdot \max(A, B, C, ...)}{4 \cdot N \cdot W_\text{ici}}$$

其中，$W_\text{ici}$ 是双向的 ICI 带宽，$N = A \cdot B \cdot C \cdot \ldots$ 是设备总数。等价地，从每个设备字节数 $V / N$ 的角度看，开销为 $(V / N) \cdot \max(A, B, C, ...) / (4 \cdot W_\text{ici})$。对于一维网格，此式简化为 $V / (4 \cdot W_\text{ici})$，即 AllGather 开销的四分之一。在二维情况下，开销实际上随最小轴的尺寸下降。

*附注：如果你想对这一结论有个大致的推导，可以从一个一维环面 $\mathbb{Z} / N\mathbb{Z}$ 开始。如果随机选择源节点和目标节点，它们之间的平均跳数约为 N / 4，由此得到开销 $(V \cdot N) / (4 * N)$。现在考虑 N 维环面，每个轴基本上是独立的。每个节点持有 $1 / N$ 字节的数据，平均需要 $\max(A, B, C, …) / 4$ 跳才能将其数据送达。你也可以从分频带宽的角度推导：在 AllToAll 中，网格的每一半将其一半的数据（$V / 4$ 字节）发送到另一半。最窄的分频带垂直于最长轴，跨越 $2 \cdot N / \max(A, B, …)$ 条链路（考虑两个切割平面，包括环绕），单向带宽为 $N \cdot W_\text{ici} / \max(A, B, …)$。两相除即得上述公式。*

### 深入探讨 ReduceScatter

ReduceScatter 是一个比其表面看起来更基础的操作，因为它实际上是 AllGather 的导数，反之亦然。即，如果在前向传播中我们有：

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

那么在反向传播中，我们将对反向模式导数 **A'**（通常在每个分片上不同）执行 ReduceScatter，以得到分片形式的 **A'**：

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

同样，如果在前向传播中是 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X]$$，那么在反向传播中就是 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$。

{% details 点击此处查看 AllGather 和 ReduceScatter 互为导数的详细说明。 %}

这源于一个事实：广播和归约作为线性算子互为转置，而 AllGather 和 ReduceScatter 分别是广播和归约的外积（也称为[克罗内克积](https://en.wikipedia.org/wiki/Kronecker_product)）。具体来说，如果有一个向量 $x \in \mathbb{R}^n$，任意数量的设备 $p \in \mathbb{N}$，并且令 $u = (1, \ldots, 1) \in \mathbb{R}^p$，我们可以如下定义广播和归约，这应与你的直观理解相符：

$$
\begin{align*}
\text{broadcast} &: \mathbb{R}^n \rightarrow \mathbb{R}^{p n} \\
\text{broadcast} &= u \otimes \mathbf{I}_n \\
\text{reduce} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^n \\
\text{reduce} &= u^T \otimes \mathbf{I}_n
\end{align*}
$$

让我们通过一个例子来看其具体形式，其中 $n = 1$, $p = 2$。如果 $x = (7)$，我们有 $$\text{broadcast}(x) = \left(\begin{pmatrix} 1 \\ 1 \end{pmatrix} \otimes \begin{pmatrix} 1 \end{pmatrix}\right) x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} x = \begin{pmatrix}  7\\  7  \end{pmatrix} \in \mathbb{R}^{p n}$$。这符合预期，即将 $\mathbb{R}^n$ 中的向量广播到 $\mathbb{R}^{pn}$。现在令 $y = (8, 9)$，我们有 $$\text{reduce}(y) = \left(\begin{pmatrix} 1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1\end{pmatrix}\right) y = \begin{pmatrix} 1 & 1  \end{pmatrix} \begin{pmatrix}  8 \\ 9  \end{pmatrix} = \begin{pmatrix}   17    \end{pmatrix}$$。这同样符合预期，即将 $\mathbb{R}^{p n}$ 中的向量归约为 $\mathbb{R}^{n}$ 中的向量。由于对于任意两个矩阵 $A$ 和 $B$，有 $(A \otimes B)^T = A^T \otimes B^T$，我们可看出 $\text{reduce} = \text{broadcast}^T$。我们通过以下外积得到 AllGather 和 ReduceScatter：

$$
\begin{align*}
\text{AllGather} &: \mathbb{R}^{p n} \rightarrow \mathbb{R}^{p^2 n} \\
\text{AllGather} &= \text{broadcast} \otimes \mathbf{I}_p \\
\text{ReduceScatter} &= \mathbb{R}^{p^2 n} \rightarrow \mathbb{R}^{p n} \\
\text{ReduceScatter} &= \text{reduce} \otimes \mathbf{I}_p
\end{align*}
$$

这里我们将 $\mathbb{R}^{p^2 n}$ 视为 $\mathbb{R}^{p \times p n}$，即每个设备对应一个 $\mathbb{R}^{p n}$ 向量。建议通过小规模例子（例如 $n = 2$, $p = 3$）来体会这些算子作为矩阵的形式。利用同样的转置性质，我们再次得到 $\text{AllGather}^T = \text{ReduceScatter}$，当然 $\text{ReduceScatter}^T = \text{AllGather}$。这种转置关系在反向传播中会出现，因为如果我们有 $y = Ax$（$A$ 是某个线性算子，如 AllGather 或 ReduceScatter），那么在反向传播中，我们将获得损失对 $y$ 的导数 $\frac{\partial L}{\partial y}$，并通过 $\frac{\partial L}{\partial x} = A^T \frac{\partial L}{\partial y}$ 得到 $\frac{\partial L}{\partial x}$。这说明了 AllGather 的导数是 ReduceScatter，反之亦然。

{% enddetails %}

将 AllReduce 转化为 AllGather 和 ReduceScatter 还有一个便利之处：我们可以将最终的 AllGather 延迟到稍后某个时刻执行。很多时候，我们并不愿意付出在所有设备上重新组装完整矩阵乘积的开销。相反，即使在结合两个具有分片收缩维度的乘数时，我们也希望保持分片状态：

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

在这种情况下，我们也可以执行 ReduceScatter 而不是 AllReduce，然后可以选择性地在稍后执行 AllGather，即：

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

注意，ReduceScatter 会*引入*一个分片维度，因此在这种情况下，它自然可以在命名的 **I** 或 **K** 维度上自由选择分片。在使用 ReduceScatter 时，我们通常需要决定*在哪个*命名维度上引入新的分片（尽管选择通常由更大的建模上下文决定）。这就是为什么我们使用语法 **ReduceScatter<sub>X,K</sub>** 来指定要分片的轴。

### 如何重叠矩阵乘法的通信与计算

正如我们在[第 1 部分](../roofline)所讨论的，我们通常假设，如果通信足够快，总能将通信与某些有用的计算重叠。本节中的集合通信通常可以与矩阵乘法计算本身重叠，但实现这一点并不简单。我们使用的算法称为**集合矩阵乘法**，最初由 [Wang 等人](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)描述。以下是展示如何实现这种重叠的简化动画：

{% include figure.liquid path="assets/img/ag_matmul.gif" caption="<b>图：</b>动画展示如何将一个分片矩阵-向量乘积与后续的 AllReduce（上述案例 3）重叠执行。一个完整的矩阵乘法由多个矩阵-向量乘积组成。" %}

简单来说，我们可以在计算矩阵一个分块的乘法的同时，开始对前一个分块进行环路归约。在某些情况下，我们还可以在批次维度或矩阵输入维度上进行分块。我们在[第 10 部分](../jax-stuff)中通过一个简单的 JAX 实现进行了演示，[Mosaic 文档](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)也提供了一个很好的 GPU 示例。我们鼓励你在某个时候实现一个这样的版本。
## What Have We Learned?
* 数组的分片（Sharding）由一个**网格（Mesh）**和一个**分片（Sharding）**规范定义，前者命名我们TPU网格的物理硬件轴，后者将网格轴名分配给数组的逻辑轴。
  * 例如，**A**[I<sub>XY</sub>, J] 描述了一个抽象数组 **A**，其第一维度沿两个网格轴X和Y进行分片。结合 Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) 或其缩写形式 Mesh({'X': 4, 'Y': 8})，这告诉我们该数组沿第一维度被分成了32个分片。

* **分片数组上的算术运算与未分片数组完全相同，除非你沿着一个分片轴执行归约（Contraction）操作**。在这种情况下，我们必须引入一些通信。我们考虑四种情形：

  1. *两个数组均未沿归约维度分片*：不需要通信。
  2. *一个数组沿归约维度分片*（或归约维度沿不同轴分片）：我们在执行操作前对其中一个输入执行AllGather操作。
  3. *两个数组在归约维度上分片方式完全相同*：我们在本地执行分片乘法，然后执行一次AllReduce或ReduceScatter操作。
  4. *两个数组沿相同的网格轴，在非归约维度上进行分片*：我们首先对其中一个输入执行AllGather操作。

* TPU大致使用**4种核心通信原语（Primitives）**：
  1. AllGather: $[A_X, B] \to [A, B]$
  2. ReduceScatter: $[A, B] \\{U_X\\} \to [A_X, B]$
  3. AllToAll: $[A, B_X] \to [A_X, B]$
  4. AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$（严格来说不是原语，因为它组合了ReduceScatter + AllGather）

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 这些操作中每一个的成本和延迟**不取决于轴的大小（只要它们是带宽受限的）**，而只取决于输入数组的大小和链路的带宽。对于单向AllGather/ReduceScatter操作：

$$T_{\text{每次AllGather或ReduceScatter的通信时间}} = \frac{\text{数据量}}{\text{带宽}} \cdot \frac{\text{轴} - 1}{\text{轴}}
\longrightarrow \frac{\text{数据量}}{\text{带宽（双向）}}$$

* AllReduce由一个ReduceScatter后跟一个AllGather组成，因此其成本是上述的2倍。AllToAll只需在环上部分传递分片，因此其成本是AllGather的¼。总结如下：

| 操作              | 描述                                                                                                 | 语法                             | 运行时间                                         |
| :---------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 沿一个轴收集分片数组的所有分片，消除一个下标。                                                       | $[A_X, B] \to [A, B]$            | 字节数 / (双向ICI带宽 * 轴数)                    |
| **ReduceScatter** | 沿一个轴对部分求和数组求和，并沿另一个轴对其进行分片（添加一个下标）。                               | $[A, B] \\{U_X\\} \to [A_X, B]$  | 与AllGather相同                                  |
| **AllReduce**     | 沿一个轴对部分求和数组求和。消除一个 { U<sub>x</sub> }。组合了AllGather和ReduceScatter操作。        | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 收集（复制）一个轴，并沿相同的轴对不同的维度进行分片。                                               | $[A, B_X] \to [A_X, B]$          | AllGather / 4（对于双向环）                      |
## Some Problems to Work
*以下是基于本节内容的一些指导性问题。我们暂时不会提供所有答案，但会在后续尽可能补充。*

**问题 1 [复制分片]**：一个数组被分片 $A[I_X, J, K, \ldots]$（即仅沿 $X$ 轴分片），网格为 `Mesh({'X': 4, 'Y': 8, 'Z': 2})`。所有芯片上 $A$ 占用的总字节数与单份数组大小的比值是多少？

{% details 点击查看答案 %}

我们的数组仅沿 X 轴分片，其大小为 4，因此每个分片的实际大小为 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$。由于我们的数组在 Y 和 Z 轴上是复制的，总大小为 $Y \cdot Z \cdot \text{sizeof}(A)$，因此总大小与单芯片大小的比值为 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$。

{% enddetails %}

**问题 2 [AllGather 延迟]**：在 TPU v4p 4x4x4 分片上，网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})`，$B=1024$ 且 $D=4096$ 使用 bfloat16 格式时，$\text{AllGather}_X([B_X, D_Y])$ 需要多长时间？$$\text{AllGather}_{XY}([B_X, D_Y])$$ 需要多长时间？$$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$ 又需要多长时间？

{% details 点击查看答案 %}

由于我们拥有完整的 `4x4x4` 立方体，所有轴上都有环形链路，因此我们有 9e10 双向带宽可供使用。

1. 因为只在一个轴上收集，另一个轴是分片的，所以我们实际上是在 1 个轴上收集 $2BD / Y$ 字节。*如果你考虑沿 Y 轴的单个分片，沿 X 的 AllGather 看起来就像是对 1/Y 字节的未分片 AllGather。* 由于 TPU v4p 的 ICI 带宽为 9e10 字节/秒双向，这将花费 $2BD / (\text{9e10} \cdot Y) = 2 \cdot 1024 \cdot 4096 / (\text{9e10} \cdot 4) = 23 \mu s$。

2. 我们拥有比之前多一倍的带宽，但我们在 AllGather 整个数组，所以 `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`。这远低于 4us 的延迟下限（每跳 1us），所以我们没问题。

3. AllReduce 的成本是 AllGather 的两倍。每个分片的大小为 $2BD / (X * Y)$，因此成本约为 $4BD / (X * Y * W)$，或者大约 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`。

{% enddetails %}

**问题 3 [延迟受限的 AllGather]**：假设我们正在执行 $\text{AllGather}_X([B_X])$，但 $B$ 非常小（例如 128）。在 TPU v4p 4x4x4 分片上，网格为 `Mesh({'X': 4, 'Y': 4, 'Z': 4})`，使用 bfloat16 格式时，这应该需要多长时间？*提示：你可能处于延迟受限状态。*

{% details 点击查看答案 %}

我们的 bfloat16 数组总共仅使用 256 字节，每台设备 64 字节。由于我们在 TPU v4p 上有一个大小为 4 的轴，我们有一个环形链路，因此我们可以双向发送数组。使用 `4.5e10` 的单向带宽，每跳大约需要 `64 / 4.5e10 ~ 0`，所以我们肯定处于延迟受限状态。计算跳数，我们可以在仅 2 跳内完成整个收集，因此大约 2us 是一个很好的估计。

{% enddetails %}

**问题 4 [矩阵乘法策略]**：为了执行 $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$，在本节中我们告诉你要执行 $\text{AllGather}_X(Y[D_X, F])$ 并将完全复制的矩阵相乘（情况 2，*策略 1*）。相反，你可以将局部块相乘，如 $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$（情况 3，*策略 2*），然后执行 $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$。这两种方法各自执行多少次 FLOPs 和通信操作？哪种更好，为什么？

{% details 点击查看答案 %}

让我们从基准（*策略 1*）开始。正如我们所示，AllGather 的成本是 $2DF / W_\text{ici}$。一旦我们拥有了完全复制的数组，总计算时间为 $2BDF / C$（其中 $C$ 是我们的加速器 FLOPs/s，因为每个 TPU 执行相同的 FLOPs）。所以我们有

$$T_\text{total (Strategy 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

相比之下，新策略（策略 2）对 $2BF$ 字节执行 AllReduce，其成本为 $4BF / W_\text{ici}$，但 FLOPs 减少 $1 / X$（因为计算是分片的）。这意味着我们执行 $2\cdot B\cdot D\cdot F / X$ 次 FLOPs，并且产生的 AllReduce 在 bfloat16 中通信 $$2 \cdot 2 \cdot B \cdot F$$ 字节。因此，*策略 2* 的总时间（无 AllGather，只有后续的 AllReduce）大致为

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

问题是：*哪一个更大？* 当 $D / (X \cdot C) > 2 / W_\text{ici}$，或 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$ 时，策略 (2) 是计算受限的。我们可能合理地期望 $D \approx 8k$，所以这意味着大约 $X < 2$，这不太可能——因此我们基本上总是使用策略 2 时通信受限。使用基准（策略 1），当 $$B < C / W_\text{ici} = 2550$$ 时，我们是通信受限的，这经常发生但并非总是如此。

所以如果 $B < 2550$，我们在两种情况下都是通信受限的，我们有

$$T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

当 $D > 2B$ 且 $2B < 5100$ 时，这成立。这通常成立，所以如果我们的批次较小，策略 2 有时可能更好。当我们的批次较大（$B > 2550$）时，我们有

$$T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

当 $2 / W_\text{ici} < D / C$，或 $D > 2 * 2550 = 5100$ 时，这成立，这对于大型模型通常成立。所以这种替代策略对于大型模型通常更好，除非 $D$ 很小。

*为什么我们不总是这样做？* 嗯，实际上我们有时可能会这样做，但通常很少有一个矩阵乘法输入的收缩维度沿另一个输入未分片的轴分片。例如，如果我们正在执行 FSDP（在 [第 5 节](../training) 中解释），我们将在数据维度上分片我们的参数，但我们的激活也将沿数据分片。所以从这个意义上说，这种情况很少出现。

{% enddetails %}

**问题 5 [最小延迟]**：假设我想在 TPU v4p 4x4x4 上执行矩阵乘法 $A[I, J] \cdot_J B[J, K] \to C[I, K]$，并实现尽可能低的延迟。假设输入可以任意分片，但结果应该是完全复制的。我的输入应该如何分片？总 FLOPs 和通信时间是多少？

{% details 点击查看（部分）答案 %}

我们不会在这里提供完整的答案，但我们将从描述四个最可能的选项开始：

1. $A[I_{XYZ}, J] \cdot B[J, K]$ + 最后 AllGather
2. $A[I, J] \cdot B[J, K_{XYZ}]$ + 最后 AllGather
3. $A[I, J_{XYZ}] \cdot B[J_{XYZ}, K]$ + 最后 AllReduce
4. $A[I, J] \cdot B[J, K]$（完全复制）

我们也可以考虑沿不同网格轴对不同轴进行分片，但这不太可能改变最终成本。对于除 (4) 以外的所有情况，每个 TPU 的总 FLOPs 是相同的，但通信对于每种情况都是不同的。然后我们只需要计算每种情况的通信成本，看看哪种最低。简而言之，(1) 和 (2) 同样好。

{% enddetails %}

**问题 6：** 假设我们想在 TPU v5e 4x4 上执行 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$。我们执行什么通信操作？在通信与计算上花费多少时间？

* $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$ 呢？这是训练中最标准的场景，我们结合了数据、张量和 ZeRO 分片。
* $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$ 呢？这是推理中的标准设置，我们执行纯张量并行（+数据）。

**问题 7：** 一个典型的 Transformer 块有两个矩阵 $W_\text{in}[D, F]$ 和 $W_\text{out}[F, D]$，其中 $F \gg D$。假设我们有批次大小 B。那么整个块是 $In[B, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D]$。让我们选择 $D=8192$，$F=32768$，$B=128$，并假设一切都使用 bfloat16 格式。假设我们在 TPU v5e 2x2 分片上运行，但让我们假设每个 TPU 只有 300MB 的可用内存。In、$W_\text{in}$、$W_\text{out}$ 和 Out 应该如何分片以保持在内存限制以下，同时最小化总时间？在通信和 FLOPs 上花费多少时间？*提示：最终输出不需要完全复制，但应与输入分片相同，以便“层”可以重复。*

{% details 点击查看（部分）答案 %}

首先考虑内存。我们的两个大矩阵各使用 `2 * 8192 * 32768 = 536MB`。我们的激活 `In` 大小为 `2 * 128 * 8192 = 2MB`（足够小，无需担心）。由于每个设备只有 300MB 的备用内存，我们显然需要对矩阵乘法进行分片。

1. $In[B_X, D] * W_\text{in}[D_{XY}, F] * W_\text{out}[F, D_{XY}] \rightarrow Out[B_X, D]$（这通常称为 FSDP）
2. $In[B, D_{XY}] * W_\text{in}[D, F_{XY}] * W_\text{out}[F_{XY}, D] \rightarrow Out[B, D_{XY}]$（这称为张量并行）

第一种情况相当糟糕，因为我们首先需要对我们的大权重或激活进行 AllGather。第二种情况在开始时需要 AllGather，在结束时需要 ReduceScatter（比 AllReduce 便宜）。我将剩余的计算留作练习。

{% enddetails %}

**问题 8 [挑战]**：使用上面的简短代码片段作为模板，使用 pmap 或 shard_map 分配一个分片数组并基准测试 4 个主要通信原语（AllGather、AllReduce、ReduceScatter 和 AllToAll）。你需要使用 `jax.lax.all_gather`、`jax.lax.psum`、`jax.lax.psum_scatter` 和 `jax.lax.all_to_all`。你理解这些函数的语义吗？它们需要多长时间？

**问题 9 [分片矩阵乘法的另一种策略？]**：[上面](#case-2-one-multiplicand-has-a-sharded-contracting-dimension)我们声称，当矩阵乘法只有一个输入沿其收缩维度分片时，我们应该 AllGather 分片的矩阵并在本地执行结果收缩。你可能想到的另一种策略是执行分片矩阵乘法，然后对结果进行 AllReduce（就像两个输入都沿收缩维度分片一样），即通过以下方式实现 $A[I, J_X] *_J B[J, K] \to C[I, K]$：

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

回答以下问题：

1. 使用索引明确写出此算法用于矩阵 $A[N, M]$ 和 $B[M, K]$，精确显示在哪个设备上执行什么计算。假设 $A$ 在 ND 个设备上分片为 $A[I, J_X]$，并且你希望你的输出在所有设备上是复制的。
2. 现在假设你对最终结果不是在每个设备上复制，而是分片（跨 N 或 K 维度）感到满意。上述算法将如何改变？
3. 纯粹从上述策略（第 2 部分，而不是第 1 部分）的通信成本来看，这种通信成本与我们首先 AllGather A 然后执行矩阵乘法的算法的通信成本相比如何？

{% details 点击查看答案 %}


1. 首先计算外积，将结果存储在 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$。请注意，重复的索引不是被收缩的那个，因为我们正在执行外积。这里的求和范围跨越我们正在使用的特定设备上存储的 i 值集合。例如，如果我们有一个大小为 16 的收缩轴和 4 个设备，那么在设备 0 上，i 的范围是 {0, 1, 2, 3}；在设备 1 上，i 的范围是 {4, 5, 6, 7}；在设备 2 上，i 的范围是 {8, 9, 10, 11}；在设备 3 上，i 的范围是 {12, 13, 14, 15}。然后对存在于每个设备上的 $O[N, K]$ 的部分和执行 AllReduce，以形成完整的 $O[N, K]$。
2. 我们可以在步骤 2 中不执行 AllReduce，而是执行更便宜的 ReduceScatter，沿任一轴：$[N, K] \\{ U_X \\} \to [N_X, K]$ 或 $[N, K] \\{ U_X \\} \to [N, K_X]$。
3. 如上文主要文本所述，执行 AllGather 的成本（当我们受吞吐量限制时）与 ReduceScatter 的成本相同；它仅由我们正在处理的完整矩阵的大小决定。因此，在 gather-then-matmul 算法中，这扩展为 $NM$（因为我们在 $\text{AllGather}$ $A$）；在 matmul-then-reduce-scatter 算法中，这扩展为 NK（因为我们在 reduce-scatter $O$）。因此，两种算法的通信成本