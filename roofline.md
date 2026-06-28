---
layout: distill
title: "All About Rooflines"
# permalink: /main/
description: "When we run algorithms on hardware, we're bounded by three things: how fast our computer can do math (OPs/second), the bandwidth available for moving data around (bytes/second), and the total memory available to store data (bytes). These \"roofline\" constraints let us upper and lower bound the time of a given computation."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: ".."
previous_section_name: "Part 0: Introduction"

next_section_url: ../tpus
next_section_name: "Part 2: TPUs"

bibliography: main.bib

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

  - name: Where Does the Time Go?
  - subsections:
    - name: "Visualizing rooflines"
    - name: "Matrix multiplication"
    - name: "Network communication rooflines"
  - name: A Few Problems to Work

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

## Where Does the Time Go?
让我们从一个极其简单的问题开始：*为什么一个算法耗时50毫秒而不是50秒或5毫秒*？模型内部究竟发生了什么导致耗时较长，我们又应该预期它耗时多久？

**计算：** 深度学习模型本质上是一系列矩阵乘法，每次乘法都由浮点数乘法和加法“运算”（FLOPs）组成。加速器速度决定了这些运算需要多长时间：

$$\begin{equation}
T_\text{math} = \frac{\text{计算 FLOPs}}{\text{加速器 FLOPs/s}}
\end{equation}$$

例如，NVIDIA H100 能以 bfloat16<d-footnote>bf16 是 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a> 的缩写，这是一种常用于机器学习的16位浮点格式。</d-footnote> 格式执行约 9.89e14 FLOPs/s，而 TPU v6e 能执行 9.1e14 FLOPs/s。<d-footnote>H100 和 B200 通常只能达到标称峰值 FLOPs 的 80-85% 左右，而 TPU 在正常使用中可以接近 95%。</d-footnote> 这意味着在 H100 上执行 1e12 FLOPs 需要（大致）`1e12 / 9.89e14 = 1.01 毫秒`，在 TPU v6e 上则需要 `1e12 / 9.1e14 = 1.1 毫秒`。<d-footnote>请注意，这些芯片定价不同，此比较未进行成本归一化。</d-footnote>

**芯片内部通信：** *在加速器内部*，张量需要在加速器存储器（HBM）和计算核心之间传输。你会看到此链路的带宽被称为“HBM 带宽”。<d-footnote>NVIDIA 也称之为“存储器带宽”。</d-footnote> 在 H100 上，[大约为 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)，在 TPU v6e 上 [大约为 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)。

**芯片间通信：** 当我们将模型分布到多个加速器上时，张量需要在它们之间频繁传输。我们的硬件通常为此提供几种选择（ICI、DCN 和 PCIe），每种都有不同的带宽。

无论通信是在芯片内部还是芯片之间，我们都以字节/秒为单位进行测量，并使用以下公式估算总通信时间：

$$\begin{equation}
T_\text{comms} = \frac{\text{通信字节数}}{\text{网络/存储器带宽（字节/s）}}
\end{equation}$$

通常（但并非总是），单个芯片内的计算可以与芯片内以及芯片间的通信重叠。这意味着**我们可以通过取计算时间和通信时间的最大值来给出训练和推理时间的下界**。我们也可以**通过它们的和来给出上界**。实际上，我们针对最大值进行优化，因为代数更简单，并且通过重叠通信和计算通常可以接近这个下界。如果我们针对最大值进行优化，那么下界和上界最多相差2倍，因为 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$。然后，我们通过建模“重叠区域”和开销来进一步提高准确性，这可以通过分析你特定的模型和目标系统来获得。

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

如果我们假设通信和计算可以完美重叠，当 $T_\text{math} > T_\text{comms}$ 时，我们实现了硬件的充分利用。我们称之为“计算受限（compute-bound）”。当 $T_\text{comms} > T_\text{math}$ 时，我们倾向于“通信受限（communication-bound）”<d-footnote>在本书中，我们将交替使用“communication-bound”、“comms-bound”、“memory-bound”和“bandwidth-bound”。</d-footnote>，并且至少有一部分加速器 FLOPs/s 在等待数据传递时被浪费了。判断一个操作是计算受限还是通信受限的一种方法是查看其“**算术强度（arithmetic intensity）**”或“**操作强度（operational intensity）**”。

**定义：** 算法的算术强度由其执行的总 FLOPs 与其需要通信的字节数之比给出——无论是在芯片内部还是芯片之间。

$$\begin{equation}
\text{算术强度} = \frac{\text{计算 FLOPs}}{\text{通信字节数}}
\end{equation}$$

算术强度衡量的是给定操作的“每字节 FLOPs 数”。一阶近似下，当我们的算术强度较高时，$T_\text{math}$ 相对于 $T_\text{comms}$ 较大，我们通常能利用大部分可用 FLOPs。反之则相反，我们在通信上花费更多时间并浪费 FLOPs。这个交叉点就是我们硬件的“峰值算术强度”，即加速器峰值 FLOPs/s 与加速器带宽之比。

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{计算 FLOPs}} {\text{加速器 FLOPs/s}} > \frac{\text{通信字节数}}{\text{带宽（字节/s）}} & \\[0.5em]
\Leftrightarrow \frac{\text{计算 FLOPs}}{\text{通信字节数}} > \frac{\text{加速器 FLOPs/s}}{\text{带宽（字节/s）}} & \\[0.5em]
\Leftrightarrow \text{强度}(\text{计算}) > \text{强度}(\text{加速器}) & \\
\end{align*}$$

量 $\text{强度}(\text{加速器})$ 是加速器达到其峰值 FLOPs/s 时的算术强度。**对于 TPU v5e MXU，这个值大约是 240 FLOPs/字节**，因为该 TPU 能执行 `1.97e14` FLOPs/s，并能从 HBM 加载 `8.2e11` 字节/秒。<d-footnote>MXU 是 TPU 上的矩阵乘法单元。这里我们指定它是因为 TPU 还有其他加速器，如负责元素操作的 VPU，其峰值 FLOPs/s 不同。</d-footnote> 这意味着如果一个算法的算术强度低于 240 FLOPs/字节，它将受限于字节加载，因此我们无法充分利用硬件。<d-footnote>这仅在算法从 HBM 加载其权重并在 MXU 中运行时才成立。正如我们将在下一节讨论的，我们有时可以将参数存储在具有更高带宽的 VMEM 中。许多算法也在 VPU 中运行，其性能特征不同。</d-footnote> 让我们看一个这样的例子：

**<span style="color:#7ab5ff">示例（点积）</span>：** 要计算两个 bfloat16 精度向量的点积，`x • y: bf16[N], bf16[N] → bf16[1]`，我们需要从存储器加载 $x$ 和 $y$，每个 $2 * N = 2N$ 字节，执行 $N$ 次乘法和 $N-1$ 次加法，并将 $2$ 字节写回 HBM。
$$\begin{equation}
\text{强度}(\text{点积}) = \frac{\text{总 FLOPs}}{\text{总字节数}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

当 $N\rightarrow\infty$ 时。因此点积的算术强度为 $\frac{1}{2}$，换言之，点积每加载一个字节执行 0.5 次浮点运算。这意味着我们的算术强度低于硬件的算术强度，我们将是通信受限的。<d-footnote>上面的 240 这个数字在这里不是正确的比较对象，因为正如你将在下一节看到的，点积是在 VPU 上执行的，而不是 MXU。TPU v5p VPU 每核大约能执行 7e12 FLOPs/秒，因此其临界强度约为 3，这意味着在这里我们仍然有些通信受限。无论如何，我们的强度低且恒定这一事实意味着在大多数硬件上很难达到计算受限。</d-footnote>

### 可视化屋顶线

我们可以使用**屋顶线图（roofline plot）** 来可视化存储器和计算之间的权衡关系，该图将算法在硬件上可达到的峰值 FLOPs/s（吞吐量，y轴）与该算法的算术强度（x轴）绘制在一起。这是一个对数坐标图示例：

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>图：</b> 一个示例屋顶线图，展示了两种具有不同算术强度（算法 1 和算法 2）的算法，以及它们在不同带宽（BW1 和 BW2）下的理论峰值吞吐量。在红色区域，算法在两种带宽下都是带宽受限的，并且浪费了硬件峰值 FLOPs/s 的一部分。黄色区域仅在较低带宽（BW1）下是带宽受限的。绿色区域在所有带宽下都是计算受限的。在这里，我们使用了加速器的峰值 FLOPs/s，增加带宽或提高强度不会带来收益。" %}

上图中，随着强度从左到右增加，我们首先看到算法性能（FLOPs/s）呈线性增长，直到达到硬件的临界算术强度，对于 TPU v5e 是 240。任何强度低于此值的算法都将是带宽（BW）受限的，并受限于峰值存储器带宽（红色部分显示）。任何位于右侧的算法将充分利用我们的 FLOPs（绿色部分显示）。这里，算法 1 是通信受限的，仅使用了总硬件 FLOPs/s 的一部分。算法 2 是计算受限的。我们通常可以通过提高算法的算术强度或增加可用的存储器带宽（从 BW1 移动到 BW2）来提高算法性能。

### 矩阵乘法

让我们看看我们即将最喜欢的算法：矩阵乘法（也称为 matmul）。我们写作 $X * Y \rightarrow Z$，其中 $X$ 形状为 $\text{bf16}[B, D]$，$Y$ 形状为 $\text{bf16}[D, F]$，$Z$ 形状为 $\text{bf16}[B, F]$。执行此矩阵乘法需要加载 $2DF + 2BD$ 字节，执行 $2BDF$ FLOPs，并将 $2BF$ 字节写回。<d-footnote>技术上我们执行 $BF \times (2D - 1)$ FLOPs，但这足够接近。这来源于 $BDF$ 次乘法和 $BF * (D-1)$ 次加法。第 4 节有更多细节。</d-footnote> <d-footnote>尽管矩阵乘法的输出技术上是 float32，但我们通常在复制回 HBM 之前向下转换为 bfloat16。</d-footnote> 因此：

$$\begin{equation}
\text{强度}(\text{矩阵乘法}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

如果我们假设“批大小” $B$ 相对于 $D$ 和 $F$ 较小，可以得到一个很好的简化。那么我们得到

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approx \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{强度}(\text{矩阵乘法}) > \text{强度}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

对于 Transformer 矩阵乘法来说，这是一个合理的假设，因为我们通常有一个局部的（每个副本的）批大小 $B < 1024$ 个 token（*不是序列*），但 $D$ 和 $F > 8000$。因此，当我们的每副本<d-footnote>我们说每副本是因为，如果我们进行某种模型分片以增加用于矩阵乘法的芯片数量，我们会按相同比例扩展可用的计算和存储器带宽。因此，临界批大小对于模型权重的每个独立副本都是成立的。</d-footnote> 批大小大于 240 个 token 时，我们通常会变成计算受限，这是一个非常简单的规则！

<p markdown=1 class="takeaway">**要点：** 要使 bfloat16 矩阵乘法在大多数 TPU 上计算受限，我们需要我们的每副本 token 批大小大于 240。<d-footnote>请注意，这_不是_通常意义上的批大小（指序列批大小）。事实上，大多数屋顶线图仅取决于 token 数量，无论它们属于相同还是不同的序列。例如，如果你在 2048 个 GPU 上有一个批大小为 512 个序列、每个序列 4096 个 token，那么你有一个总批大小 `512 * 4096 = 2M` 个 token，和一个局部批大小 1k 个 token。</d-footnote></p>

这附带了一些值得注意的注意事项，我们将在下面的问题中探讨，特别是关于量化（例如，如果我们量化激活但仍执行全精度 FLOPs），但这是一个值得记住的好规则。对于 GPU，这个数字略高（接近 300），但通常结论相同。当我们[将一个大矩阵乘法分解为更小的矩阵乘法](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)时，分块大小（tile sizes）也很重要。<d-footnote>当我们执行一个大矩阵乘法时，我们需要将其分解为适合 VMEM/SMEM/TMEM（更高带宽的片上存储器）的小块。这导致我们需要多次加载数据块，因此说我们只加载 $O(N^2)$ 字节不再完全正确。考虑一个 $(m, k) \cdot (k, n)$ 的矩阵乘法，其分块大小为 $bm$, $bk$, $bn$。设 $tm = m / bm$，等等。那么总 FLOPs 为 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bn \cdot bk$，总字节数为 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + bm \cdot bn)$。忽略最后一项，我们得到强度为 $bm \cdot bn / (bm + bn)$，这与上面的类似。</d-footnote> 我们将在[下一节](../tpus)中讨论更低层次的 GPU 和 TPU 细节。

### 网络通信屋顶线

我们到目前为止讨论的所有屋顶线都是存储器带宽屋顶线，_且都在单个芯片内_。这不应被视为规则。事实上，我们在本书中关注的大多数屋顶线都涉及芯片间的通信：通常是涉及分布在多个 TPU 上的矩阵的矩阵乘法。

选择一个稍微人为构造的例子，假设我们要相乘两个大矩阵 $X\sim \text{bf16}[B, D]$ 和 $Y \sim \text{bf16}[D, F]$，它们均匀分布在 2 个 TPU/GPU 上（沿着 $D$ 维度）。为了执行这个乘法（正如我们将在[第 3 节](../sharding)中看到的），我们可以在每个 TPU 上相乘每个矩阵的一半（在 TPU 0 上执行 `Z0 = X[:, :D // 2] @ Y[:D // 2, :]`，在 TPU 1 上执行 `Z1 = X[:, D // 2:] @ Y[D // 2:, :]`），然后将得到的“部分和”复制到另一个 TPU 并相加。假设我们可以每个方向复制 `4.5e10` 字节/秒，并且在每个芯片上执行 `1.97e14`
## A Few Problems to Work
**问题 1 [int8 矩阵乘法]:** 假设我们想要执行矩阵乘法 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$<d-footnote>此处及下文中，我们将使用 $A \cdot_D B$ 的表示法来表示乘法在 D 维度上进行收缩。这是对爱因斯坦求和（einsum）表示法的滥用。</d-footnote>，采用 int8 精度（每个参数 1 字节），而不是 bfloat16（每个参数 2 字节），因为 TPU/GPU 在低精度下能更快地执行矩阵乘法。

1.  需要从内存中加载多少字节？需要写回多少字节？
2.  总共执行了多少次运算（OPs）？
3.  算术强度（arithmetic intensity）是多少？
4.  对 $T_\text{math}$ 和 $T_\text{comms}$ 的屋顶线（roofline）估计是什么？整个操作运行时间的合理上限和下限是多少？

假设我们的 HBM 带宽为 `8.2e11` 字节/秒，我们的 int8 峰值运算次数为 `3.94e14` 次/秒（大约是 bfloat16 的 2 倍）。

{% details 点击这里查看答案。 %}

1.  由于我们以 int8 存储参数，每个参数占 1 字节，因此我们从 HBM 加载 $$BD + DF$$ 字节，并写回 $$BF$$ 字节。
2.  这与 bfloat16 情况相同，但理论上 int8 的运算次数/秒应该更快。因此总运算次数仍然是 $2BDF$。
3.  算术强度是 $$2BDF / (BD + DF + BF)$$。如果我们像之前一样假设 $$B \ll D$$ 且 $$B \ll F$$，我们得到的算术强度为 $$2B$$，这意味着我们的规则变为 $B > \text{HBM int8 算术强度} / 2$。使用给定的数字，这个 int8 强度为 `3.94e14 / 8.2e11 = 480`，所以规则是 $B > 480 / 2 = 240$。请注意，这基本没有变化！
4.  $$T_\text{math} = 2BDF / 3.94e14$$ 且 $$T_\text{comms} = (BD + DF + BF) / 8.2e11$$，因此一个合理的下界是 $$\max(T_\text{math}, T_\text{comms})$$，上界是 $$T_\text{math} + T_\text{comms}$$。

{% enddetails %}

**问题 2 [int8 + bf16 矩阵乘法]:** 在实践中，我们经常对权重（weight）和激活值（activation）采用不同的量化方案，因此我们可能以非常低的精度存储权重，但将激活值（和计算）保持在较高的精度。假设我们想要将权重量化为 int8，但将激活值（和计算）保持在 bfloat16。在什么批大小（batch size）下我们会变成计算受限（compute bound）？假设 `1.97e14` bfloat16 FLOPs/秒。

*提示：这具体意味着 `bf16[B, D] * int8[D, F] -> bf16[B, F]`，其中 $B$ 是“批大小”。*

{% details 点击这里查看答案。 %}

再次假设 B 很小，我们有 2BDF bfloat16 次浮点运算，但只有 DF 权重（而不是 bfloat16 的 2DF）。这意味着当 $$2B > 240$$ 或 $$B > 120$$ 时，我们会变成计算受限。这个值低得多，这意味着如果我们能进行 int8 权重量化（这相当容易）但仍然执行 bfloat16 浮点运算，我们会在效率上获得显著提升（尽管 int8 运算会更好）。

{% enddetails %}

**问题 3:** 接续问题 2 的设置，对于 $F = D = 4096$ 和 $F = D = 1024$，绘制峰值 FLOPs/秒关于 $B$ 的屋顶线图。*请使用加载的精确字节数，而非近似值。*

{% details 点击这里查看答案。 %}

这是所讨论的图表：

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

注意，两个模型最终都达到了硬件峰值 FLOPs/秒，但更大的 D/F 更早达到。D=F=1024 几乎使临界批大小（critical batch size）翻倍。生成此图的代码在此：

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**问题 4:** 如果我们想执行 $\text{int8}[B, D] \cdot_D \text{int8}[B, D, F] \rightarrow \text{int8}[B, F]$，其中我们设想为每个批元素使用一个不同的矩阵。这个操作的算术强度是多少？

{% details 点击这里查看答案。 %}

让我们先看看总浮点运算次数和通信量。

1.  总浮点运算次数：运算次数基本相同，因为我们正在执行 $$B$$ 个独立的 $$[D] \times [D, F]$$ 乘积，其总工作量与单个 $$[B, D] \times [D, F]$$ 矩阵乘法相同（这在第 4 节中有更多讨论）。所以这只是 $$2BDF$$。
2.  总通信量：这里通信量要大得多：$$BD + BDF + BF$$。
3.  因此，我们的算术强度现在实际上是 $$2BDF / (BD + BDF + BF)$$。由于 $$BDF$$ 在分母中占主导地位，这大约是 $$2$$。所以，它不再是依赖于批大小，而是本质上为常数。这很糟糕，因为这意味着无论批大小如何，我们基本上总是通信受限。

{% enddetails %}

**问题 5 [GPU 的内存屋顶线]:** 使用 [NVIDIA 为 H100 SXM 提供的规格说明](https://www.nvidia.com/en-us/data-center/h100/)，计算 bfloat16 矩阵乘法在什么批大小下会变成计算受限。*请注意，Tensor Core 的 FLOPs 数字是真实值的两倍，因为它们仅通过结构化稀疏性才能达到。*

{% details 点击这里查看答案。 %}

从规格说明中，我们看到报告的 bfloat16 FLOPs 值为 `1.979e15` FLOPs/秒，带有星号注明“启用稀疏性”。真实值在没有稀疏性时是它的一半，即 `9.89e14` FLOPs/秒。内存带宽为 3.35TB/s，或 `3.35e12` 字节/秒。因此 $B_\text{crit}$ 为 `9.89e14 / 3.35e12 = 295`，与 TPU 的结果非常相似。

{% enddetails %}

<h3 markdown=1 class="next-section">第一部分到此结束！对于第二部分，了解真实的 TPU 如何处理浮点运算和通信，请[点击这里](../tpus)。</h3>