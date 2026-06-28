---
layout: distill
title: "How to Think About TPUs"
# permalink: /main/
description: "This section is all about how TPUs work, how they're networked together to enable multi-chip training and inference, and how this affects the performance of our favorite algorithms. There's even some good stuff for GPU users too!"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "Part 1: Rooflines"

next_section_url: ../sharding
next_section_name: "Part 3: Sharding"

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
  - name: What Is a TPU?
  - name: TPU Networking
  - name: Key Takeaways
  - subsections:
    - name: TPU specs
  - name: Worked Problems
  - name: Appendix
  - subsections:
    - name: "Appendix A: More on TPU internals"
    - name: "Appendix B: How does a systolic array work?"

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
---<p markdown=1 class="announce">您可能也会感兴趣阅读关于NVIDIA GPU的新[第12章](../gpus)！</p>
## What Is a TPU?
TPU本质上是一个专注于矩阵乘法（称为TensorCore）的计算核心，连接着高速存储器堆栈（称为高带宽存储器或HBM）<d-cite key="tpu_paper"></d-cite>。以下是示意图：

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>图：</b>TPU芯片的基本组件。TensorCore是左侧灰色方框，内含矩阵乘法单元（MXU）、向量处理单元（VPU）和向量存储器（VMEM）。" %}

可以将TensorCore理解为一台性能卓越的矩阵乘法专用机，但它还具有一些值得注意的其他功能。TensorCore包含三个核心单元：

* **MXU**（矩阵乘法单元）是TensorCore的核心。对于大多数TPU代次，它每8个时钟周期执行一次`bf16[8,128] @ bf16[128,128] -> f32[8,128]`矩阵乘法<d-footnote>TPU v6e（Trillium）采用256x256的MXU，而此前所有代次均为128x128。</d-footnote>，使用脉动阵列（详见<a href="#appendix-b-how-does-a-systolic-array-work">附录B</a>）。
  * 以TPU v5e的1.5GHz主频计算，每个MXU可实现约`5e13` bf16 FLOPs/s。大多数TensorCore配备2或4个MXU，例如TPU v5e的总bf16 FLOPs/s为`2e14`。
  * TPU还支持更高吞吐量的低精度矩阵乘法（例如每个TPU v5e芯片可实现`4e14` int8 OPs/s）。

* **VPU**（向量处理单元）执行通用数学运算，如ReLU激活函数或向量间的逐元素加法/乘法，归约运算（如求和）也在此完成。<a href="#appendix-a-more-on-tpu-internals">附录A</a>提供更详细说明。
* **VMEM**（向量存储器）是TensorCore内的片上暂存器，紧邻计算单元。其容量远小于HBM（例如TPU v5e为128 MiB），但对MXU的带宽高得多。VMEM的工作方式类似CPU的L1/L2缓存，但容量更大且由程序员控制。HBM中的数据需先复制到VMEM才能进行计算。

**TPU在矩阵乘法方面极其高效**。这既是其核心功能也是其优势所在。[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)作为迄今最强大的TPU之一，单核可实现`2.5e14` bf16 FLOPs/秒，单芯片可达`5e14` bf16 FLOPs/秒。由8960个芯片组成的Pod集群可实现4 exaFLOPs/s的bf16算力，这相当于世界顶级超级计算机的性能规模。<d-footnote>TPU及其脉动阵列能成为强大硬件加速器的关键在于：矩阵乘法是少数计算复杂度为$O(n^3)$而数据量仅为$O(n^2)$的算法之一，这使得普通ALU极易受计算能力而非内存带宽的瓶颈限制。</d-footnote>

上图还包含SMEM和标量单元等其他组件，主要用于控制流处理（详见<a href="#appendix-a-more-on-tpu-internals">附录A</a>），并非理解关键。而HBM则是重要且直观的组件：

* **HBM**（高带宽存储器）是存储张量供TensorCore使用的大型高速存储单元，容量通常达数十GB量级（例如[TPU v5e配备16GiB HBM](https://cloud.google.com/tpu/docs/v5e#system_architecture)）。
  * 计算时，张量通过VMEM从HBM流式加载至MXU，结果再通过VMEM逐块写回HBM。
  * HBM与TensorCore间（经由VMEM）的带宽称为"HBM带宽"（通常约1-2TB/秒），这决定了内存密集型工作负载的计算速度上限。

**通常所有TPU操作都采用流水线与重叠执行**。执行矩阵乘法$X \cdot A \to Y$时，TPU首先将矩阵$A$和$X$的分块从HBM复制到VMEM，随后加载至MXU进行8x128（$X$）与128x128（$A$）分块乘法，最后将结果分块写回HBM。为实现高效执行，矩阵乘法采用流水线设计，使VMEM的数据传输与MXU的计算重叠进行，从而避免MXU等待内存传输，确保计算受限而非内存受限。

以下是HBM逐元素乘法的示例动画：

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>图：</b>TPU执行逐元素乘法的动画，数据从HBM加载。注意数据如何分块从内存流出，部分结果通过流水线回写而无需等待完整数组生成。" %}

矩阵乘法的过程几乎相同，区别在于数据加载目标为MXU而非VPU/向量单元，且因权重分块需复用于多个激活分块，加载存储顺序有所不同。可以看到数据流经VMEM、VREGs（向量寄存器）、向量单元，最终返回VMEM和HBM。接下来将看到，若HBM到VMEM的加载速度低于向量单元（或MXU）的计算速度，就会因数据供给不足导致"带宽受限"。

<p markdown=1 class="takeaway">**核心要点：** TPU架构简洁高效：将权重从HBM加载至VMEM，再传入脉动阵列执行约200万亿次/秒的乘加运算。HBM$\leftrightarrow$VMEM与VMEM$\leftrightarrow$脉动阵列的带宽从根本上决定了TPU能高效执行的计算类型。</p>

**VMEM与计算强度：** VMEM容量虽远小于HBM，但对MXU的带宽更高。正如[第1节](../roofline)所述，这意味着若算法能将输入/输出全部置于VMEM，则更不易遇到通信瓶颈。这对计算强度较低的算法尤其有利：VMEM带宽约为HBM的22倍，使得MXU从VMEM读写操作时仅需10-20的计算强度即可达到峰值算力利用率。这意味着若能将权重存入VMEM而非HBM，即使批次尺寸较小，矩阵乘法也能达到算力受限状态，同时保证固有计算强度较低算法的执行效率。但VMEM容量有限，这常构成挑战。<d-footnote>我们有时讨论VMEM预加载，即提前将权重加载至VMEM以掩盖矩阵乘法的数据加载开销。例如在Transformer模型中，有时可在注意力计算期间将大型前馈权重加载至VMEM，从而在内存带宽受限时隐藏权重加载开销。这要求权重规模足够小或经过充分分片，使得单层权重能完全载入VMEM且有剩余空间。</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**TPU芯片通常（并非总是）由两个共享内存的TPU核组成**，可视为一个双倍算力的大型加速器（称为"巨核"配置）。v4、v5和v6代TPU均采用此设计（TPU v7取消巨核配置，改为高带宽互连双核）。较早代次的TPU芯片内存独立，被视为两个独立加速器（TPU v3及更早版本）。推理优化型芯片（如TPU v5e）每芯片仅含一个TPU核。

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**芯片**以**4个为一组安装在托盘**上，通过**PCIe网络连接至CPU主机**。这是多数用户熟悉的配置形式：通过Colab或单台TPU-VM可见的4芯片（8核，但通常视为4个逻辑巨核）。对于推理芯片如TPU v5e，每个主机配备2个托盘（而非1个），但每芯片仅1核，形成8芯片=8核的配置。<d-footnote>在Cloud TPU虚拟机中，每个托盘作为独立虚拟机的一部分呈现，因此仍可见4个核。</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe带宽有限：** 类似HBM$\leftrightarrow$VMEM链路，CPU$\leftrightarrow$HBM的PCIe连接具有特定带宽，限制了主机内存与HBM间的数据传输速率。例如TPU v4的PCIe带宽为单向16GB/秒，比HBM慢近100倍。虽然*可以*向主机（CPU）内存加载/卸载数据，但速度较慢。
## TPU Networking
在Pod中，芯片通过ICI网络相互连接。在旧一代产品（TPU v2和TPU v3）、推理芯片（如TPU v5e）以及Trillium（TPU v6e）中，ICI（“芯片间互连”，inter-chip interconnects）连接了4个最近的邻居（通过边缘链接形成二维环面）。TPU v4和TPU v5p则连接到最近的6个邻居（形成三维环面）。请注意，这些连接**不**经过主机，而是芯片之间的直连。

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

环面结构将任意两个节点之间的最大距离从 $N$ 减少到 $N / 2$，从而使通信速度大大加快。TPU还具有“扭曲环面”（twisted torus）配置，它将环面包裹成类似莫比乌斯带（Mobius-strip）的拓扑结构，以进一步减少节点之间的平均距离。

**由ICI连接的TPU Pod可以变得非常大：** TPU v4的最大Pod尺寸（称为**超级Pod**，superpod）为 `16x16x16`，TPU v5p为 `16x20x28`。这些大型Pod由可重构的 `4x4x4` 芯片立方体组成，它们通过[光学环绕链路](https://arxiv.org/pdf/2208.10041)<d-footnote>光学交换机本质上是一种可重构连接，具有与ICI相同的带宽。它只是允许我们在连接立方体的同时保持环绕链接。</d-footnote>连接，我们可以将其重新配置以连接非常大的拓扑结构。

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

也可以请求较小的拓扑结构（例如 `2x2x1`, `2x2x2`），但没有环绕链接。这是一个重要的注意事项，因为它通常会使大多数通信的时间翻倍。任何完整立方体（例如 `4x4x4` 或 `4x4x8`）的倍数都将由光学交换机提供环绕链接。<d-footnote>请注意，`2x2x4` 不会拥有任何环绕链接，因为环绕链接由光学交换机提供，而光学交换机仅在完整立方体上可用。然而，TPU v5e 的 8x16 拓扑_将_在较长的轴向上拥有环绕链接，因为它不使用可重构的光网络。</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e和Trillium的Pod由一个单一的 `16x16` 二维环面组成，沿任何尺寸为16的轴都有环绕链接（这意味着 `8x16` 拓扑在长轴上有环绕链接）。TPU v5e和v6e（Trillium）无法扩展到超过16x16的环面，但Pod之间仍然可以通过标准数据中心网络（DCN）相互通信，DCN将TPU主机相互连接。同样，可以请求较小的拓扑结构，但在维度 $<16$ 时没有环绕链接。

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**这种最近邻连接是TPU与GPU之间的关键区别**。GPU通过分层交换机连接，这些交换机近似于在每个GPU之间建立点对点连接，而不是像TPU那样使用本地连接。通常，一个节点内的GPU（H100有8个GPU，B200 NVL72多达72个）是直接连接的，而更大的拓扑结构则需要在每个GPU之间经过O(log(N))跳转。一方面，这意味着GPU可以在少量跳转内发送任意数据。另一方面，TPU的成本要低得多（因为NVLink交换机价格昂贵），布线更简单，并且可以扩展到更大的拓扑结构，因为每台设备的链路数量和每台设备的带宽是恒定的。在[此处](../gpus#networking)阅读更多信息。

**ICI相对于DCN非常快，但仍慢于HBM带宽。** 例如，[TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture) 具有：
*   每片芯片 `2.8e12` 字节/秒（2.8 TB/s）的HBM带宽。
*   每轴 `9e10` 字节/秒（90 GB/s）的ICI带宽，每片芯片有3个轴。<d-footnote>上文链接的页面列出了100 GB/s的带宽，与此处列出的略有不同。TPU ICI链路的带宽根据执行的操作略有不同。通常可以放心使用本文档中的数字。</d-footnote>
*   每个TPU（通过每个主机上的1-2个网卡）`6.25e9` 字节/秒（6.25 GB/s）的DCN（出口）带宽。<d-footnote>TPU v6e和TPU7x为12.5e9字节/秒，v5e为3.125e9字节/秒。</d-footnote>

这意味着，当我们将模型分布到多个芯片上时，需要小心避免较慢的跨设备通信成为MXU的瓶颈。

**多切片训练（Multi-slice training）：** 一组由ICI连接的TPU被称为一个**切片**（slice）。不同的切片可以使用DCN相互连接，例如连接不同Pod上的切片。由于DCN是比ICI慢得多的连接，我们应尽量限制计算需要等待DCN数据的时间。DCN是主机到主机的连接，因此要通过DCN在TPU之间传输缓冲区，我们首先需要通过PCIe将数据传输到主机，然后通过网络出口传输，再通过目标主机网络入口接收，最后通过PCIe传入HBM。
## Key Takeaways
* TPU结构简单，大多数情况下可将其视为一个矩阵乘法单元（matrix multiply unit），该单元通过ICI连接至超高速存储器，通过ICI连接至其他芯片（速度较快），并通过DCN连接至数据中心其他部分（速度中等）。

* 通信速度受限于各类网络带宽，按速度排序如下：
  * HBM带宽（HBM bandwidth）：TensorCore与其关联HBM之间的带宽。
  * ICI带宽（ICI bandwidth）：TPU芯片与其最近的4个或6个邻居之间的带宽。
  * PCIe带宽（PCIe bandwidth）：CPU主机与其关联的芯片板卡之间的带宽。
  * DCN带宽（DCN bandwidth）：多个CPU主机之间的带宽（通常指未通过ICI直接连接的主机）。

* **在一个切片（slice）内，TPU仅通过ICI与最近的邻居连接。** 这意味着切片内距离较远的芯片进行ICI通信时，需要经过中间芯片的逐级转发。

* **权重矩阵（weight matrices）在两个维度上都必须填充至至少128大小**（在TPU v6e上为256），以充分利用MXU的计算资源（实际上，较小的轴会填充至128）。

* **低精度矩阵乘法通常速度更快。** 对于支持的代际，TPU执行int8或int4运算的速度大约是bfloat16浮点运算（FLOPs）的2倍/4倍。向量处理单元（VPU）的操作仍以fp32精度执行。

* 为避免TPU计算单元成为瓶颈，我们需要**确保通过每个通道的通信量与其速度成比例**。

### TPU规格

以下是我们芯片的一些具体参数：

| 型号                                      | Pod规模 | 主机规模 | 每芯片HBM容量 | 每芯片HBM带宽 (字节/秒) | 每芯片浮点运算次数/秒 (bf16) | 每芯片浮点运算次数/秒 (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.2e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |
| <span class="nowrap-header">TPU7x</span>   | 4x4x576  |   2x2x1   |       192GB       |        7.4e12         |       2.30e15       |       4.61e15       |

主机规模（Host size）指的是连接到单个CPU主机的TPU的拓扑结构（例如，TPU v5e的单个CPU主机以4x2拓扑连接8个TPU）。关于最新代际的更多详情，请参阅[TPU7x文档](https://docs.cloud.google.com/tpu/docs/tpu7x)。以下是互连（interconnect）参数：

| 型号       | ICI每链路带宽 (单向，字节/秒) | ICI每链路带宽 (双向，字节/秒) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |             1.0e11             |           2.0e11            |
| **TPU v4p** |             4.5e10             |           9.0e10            |
| **TPU v5p** |             9.0e10             |           1.8e11            |
| **TPU v5e** |             4.5e10             |           9.0e10            |
| **TPU v6e** |             9.0e10             |           1.8e11            |
| **TPU7x**   |             9.0e10             |           1.8e11            |

我们同时列出了单向（单向）带宽和双向带宽（bidi bandwidth），因为单向带宽更贴近硬件实际情况，而双向带宽在涉及完整环路（full ring）的方程中出现频率更高。<d-footnote>双向带宽（bidi bandwidth）是指沿着单条链路在两个方向上可传输的总字节数，或者等效地，指单个TPU沿特定轴向外发送的总字节数（假设能有效利用两条链路）。当我们拥有一个完整的环路时——即在特定轴上存在环绕连接时——这一假设成立。对于推理芯片，当拥有完整的16轴时成立；对于训练芯片（v*p系列），当轴的大小是4的倍数时成立。我们倾向于使用双向带宽，因为它在涉及双向通信的计算中频繁出现。</d-footnote>

PCIe带宽通常约为每个TPU `1.6e10` 字节/秒（TPU v6e为 `3.2e10`），而DCN带宽通常约为每个TPU `6.25e9` 字节/秒（TPU v6e和TPU7x为 `12.5e9`，TPU v5e为 `3.125e9`）。
## Worked Problems
这些数字有些枯燥，但它们能让你对模型性能进行基本的**Roofline估计**。让我们解决几个问题来说明为什么这很有用。你将在第三部分看到更多示例。

**问题1 [限制大语言模型延迟]：** 假设你想从一个使用bf16格式的2000亿参数模型中进行采样，该模型分布在32个TPU v4p上。将所有参数从**高带宽内存**加载到**脉动阵列**需要多长时间？*提示：使用上面的数字。*

{% details 点击这里查看答案。 %}

**答案：** 我们需要在32个芯片上加载`sizeof(bf16) * 200e9 = 400e9`字节，这意味着每个芯片12.5e9字节，每个芯片的HBM带宽为1.23e12。因此加载大约需要10毫秒。

这非常酷，因为*这是从模型中进行采样的合理延迟下限*。每个采样步骤都需要从HBM加载所有参数，因此不能少于10毫秒。实际上，在小批量大小时，这接近可以实现。

{% enddetails %}

**问题2 [TPU细节]：** 考虑一个完整的TPU v5e pod。总共有多少个CPU主机？多少个TPU TensorCore？整个pod的总FLOPs/s是多少？总HBM是多少？对TPU v5p pod做同样的练习。

{% details 点击这里查看答案。 %}

**答案：** 对于TPU v5e，每个pod是`16x16`，每个主机是一个4x2切片，所以我们有`16*16 / 8 = 32`个主机。对于TPU v5e，每个TPU只有一个核心，所以我们有256个TensorCore。总FLOPs/s是`16*16*2e14 = 5.1e16`（bfloat16格式）。每个芯片有16GB的HBM，所以总内存是`256 * 16 = 4TB`。

对于一个完整的TPU v5p pod，我们有`16x20x28`个芯片，每个主机是2x2x1，所以我们有`(16*20*28) / (2*2) = 2,240`个主机。对于TPU v5p，每个TPU有两个TensorCore，所以我们有`8960 * 2 = 17,920`个核心。总FLOPs/s是`8960 * 4.59e14 = 4.1e18`（bfloat16格式）。每个芯片有96GB的HBM，所以总内存是`8960 * 96 = 860TB`。

{% enddetails %}

**问题3 [PCIe操作强度]：** 想象一下，我们被迫将一个类型为$\text{bf16}[D, F]$的大权重矩阵 $A$ 和一个类型为$\text{bf16}[B, D]$的激活批 $x$ 存储在主机**DRAM**中，并想对它们进行矩阵乘法。这运行在一个单一主机上，我们使用一个连接到它的单一TPU v6e芯片。你可以假设$B \ll D$，且$F = 4D$（我们将在后面的章节中看到为什么这些是合理的假设）。我们需要的最小批量大小 $B$ 是多少才能在PCIe上保持**计算受限**？假设PCIe带宽为1.6e10字节/秒。

{% details 点击这里查看答案。 %}

**答案：** 我们必须执行$2BDF$次浮点运算，每个芯片可以执行`9.2e14`次浮点运算每秒。这需要$2BDF / 9.2e14$秒来执行。我们必须从DRAM加载$2DF + 2BD$字节，并写回$2BF$字节。我们受限于PCIe传输速度，因此需要$2 \cdot (BD + DF + BF) / 1.6e10$秒来将数据传输到TPU和从TPU传输回来。由于我们希望计算时间比权重加载时间长，假设我们可以将所有权重加载与计算重叠，我们希望 $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.6e10$。利用我们的假设$B \ll D$和$F = 4D$，可以将其简化为

$$\frac{8BD^2}{9.2 \times 10^{14}} > \frac{8D^2}{1.6 \times 10^{10}}$$

或

$$B > \frac{9.2 \times 10^{14}}{1.6 \times 10^{10}} \simeq 57{,}500$$

{% enddetails %}

**问题4 [一般矩阵乘法延迟]：** 假设我们想将一个权重矩阵int8[16384, 4096]乘以一个大小为int8[B, 4096]的激活矩阵，其中B是某个未知的批量大小。假设我们一开始在1个TPU v5e上。

1. 这个乘法作为B的函数需要多长时间？*提示：计算从HBM加载数组需要多长时间以及乘法实际需要多长时间可能会有所帮助。哪个是你的瓶颈？*
2. 如果我们想从**VMEM**运行这个操作呢？作为B的函数需要多长时间？

{% details 点击这里查看答案。 %}

**答案：** (1) 我们需要执行的操作数是 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3 \times 10^{8} \cdot B$。所以 $T_{\text{math}} = (1.3 \times 10^{8} \cdot B) / 3.94 \times 10^{14}$ 秒。我们需要从HBM加载 $16384 \cdot 4096 + 4096 \cdot B$ 字节到VMEM，并从VMEM写回 $16384 \cdot B$ 字节到HBM。这意味着 $T_{\text{comms}} = (6.7 \times 10^{7} + 2 \times 10^{4} \cdot B) / 8.2 \times 10^{11}$ 秒。假设通信和计算尽可能多地重叠，整个乘法大约需要

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}, \frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}}\right\}$$

当 $\frac{6.7 \times 10^{7} + 2 \times 10^{4} \cdot B}{8.2 \times 10^{11}} < \frac{1.3 \times 10^{8} \cdot B}{3.94 \times 10^{14}}$，或者等效地，当 $B > 267$ 时，我们将是计算受限的。这比我们在[第1节](../roofline)中推导出的240略大，因为我们考虑了 $D$ 和 $F$ 的全部影响。

(2) 如果我们是从VMEM加载，让我们考虑VMEM到MXU的带宽是HBM $\leftrightarrow$ VMEM带宽的22倍。这将我们的数据加载分母从8.2e11改为1.80e13，我们得到 $B > 11$。请注意，在实践中，我们无法将所有VMEM带宽专用于加载权重矩阵，因此实际上它会更接近20。

{% enddetails %}

**问题5 [ICI带宽]：** 假设我们有一个TPU v5e `4x4`切片。假设我们想将一个类型为`bf16[8, 128, 8192]`的数组从`TPU{0,0}`发送到`TPU{3, 3}`。假设TPU v5e的每跳延迟为 $1\mu s$。

1. 第一个字节将在多快后到达其目的地？
2. 总传输需要多长时间？

{% details 点击这里查看答案。 %}

**答案：** 在TPU v5e中，我们有2D连接。因为我们只有一个`4x4`切片（没有大小为16的轴），我们没有回绕连接。因此，目标芯片有两个端口可以接收数据，同样源芯片也有两个端口可以发送数据。我们必须传输的数据量是`2 * 8 * 128 * 8192 = 1.7e7`字节。我们可以同时从两个端口传输（即，一半数组向右发送，一半向下发送），因此我们得到每秒`2 * 4.5e10 = 9e10`字节的传输量，这意味着传输整个数组大约需要`1.7e7 / 9e10 = 188us`（假设我们是带宽受限）。在一个`4x4`切片中，芯片$(0, 0)$和$(3, 3)$之间有六跳，因为对于少于16个芯片的轴没有回绕链接。由于每跳延迟大约为 $1\mu s$，第一个字节将在大约`6us`内到达，总传输将花费大约`188 + 6 = 194us`，因为最后一个字节在离开源后同样必须经过六跳（一般来说，延迟和带宽项是相加的，但在这里延迟是一个小的修正）。

{% enddetails %}

**问题6 [综合应用，困难]：** 想象你有一个大矩阵 **A**：`int8[128 * 1024, 128 * 1024]` 均匀地分片在一个TPU v5e 4x4切片上，但卸载到每个芯片的主机DRAM中。假设你想将整个数组复制到TPU{0, 0}，并将其乘以一个向量 `bf16[8, 128 * 1024]`。这需要多长时间？*提示：使用上面的数字。*

{% details 点击这里查看答案。 %}

**答案：** 让我们从概述我们必须执行的操作开始。我们的数组大约16GB。根据上面的表格，一个TPU v5e主机具有4x2拓扑结构，因此一个4x4有2个主机。因此，由于我们的数组是均匀分片的，每个主机实际上包含数组的1/2块，即8GB。我们需要将这些块全部复制到TPU{0,0}，这给了我们两个选择：

1. 我们可以通过**DCN**复制，然后通过PCIe将整个未分片的数组加载到HBM。
2. 我们可以将分片后的数组加载到它们对应的TPU上，然后通过ICI执行**集合通信**，然后在TPU{0,0}上执行矩阵乘法。

应该清楚的是，选项(2)更好。与ICI相比，DCN速度慢，我们更愿意通过多个PCIe链接加载一个大数组，而不是仅通过少数几个（主机0上的8个）。这里是系统部分的示意图。如上所述，请注意TPU通过ICI连接到它们的邻居（甚至跨主机），所有TPU都连接到它们的主机CPU（通过PCIe），并且主机通过DCN连接。

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="每个芯片实际上都有自己的PCIe链接到其主机，但为了清晰起见，这里只显示了一个。" %}

现在让我们计算每个部分需要多长时间：

1. **PCIe加载**：我们通过16个PCIe链接加载16GB的数据块，每个链接的带宽为`1.6e10`字节/秒。因此这将花费大约63毫秒。

2. **ICI复制**：每个TPU现在有我们数组的16GB / 16 = 1GB。我们的ICI带宽是每链路双向`9e10`字节/秒，从上图你会注意到，在这个拓扑结构中，TPU v5e上的4个ICI链接中只有2个在TPU{0,0}上使用。由于TPU{0,0}需要沿2个轴以`4.5e10`字节/秒/链路的速度接收总共15GB，我们可以通过`15e9 / (4.5e10 * 2) = 167ms`来降低时间下限。实际上这可能无法实现，因为负载非常不均匀，但它可能在2倍范围内。正如你将在第3节看到的，执行完整的AllGather也将花费大约`16e9 / (4.5e10 * 2)`，因此这接近最优。

3. **HBM $\rightarrow$ MXU加载**：为了执行我们的最终矩阵乘法，我们需要将这些16e9字节加上bf16[8, 128 \* 1024]数组（另外2MB，可以忽略）通过HBM带宽加载到MXU，这将花费`16e9 / 8.2e11 = 20ms`。

4. **FLOPs**：我们总共执行 $$2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7 \times 10^{11}$$ 次浮点运算，由于我们可以执行`1.97e14` bf16 FLOPs/s，我们得到1.4毫秒。

总时间的上限是所有这些时间的总和，但由于TPU通常可以重叠这些操作，我们可以将其视为一个受最慢部分限制的流水线问题。假设这是真的，那么答案至少是167毫秒，在重叠不完善的情况下可能接近200毫秒。

{% enddetails %}

<h3 markdown=1 class="next-section">第二部分到此结束！关于涵盖分片和跨TPU通信的第三部分，请[点击这里](../sharding)。</h3>
## Appendix
### 附录 A：深入探讨TPU内部构造

本节将更深入地探讨TPU的内部工作原理。除特别说明外，我们将以TPU v5p的规格为基准进行介绍。

### 向量处理单元 (VPU)

VPU是TPU的向量算术核心。它由一个二维SIMD向量机（即**VPU**本身，执行vadd（向量加法）或vmax（元素级最大值）等元素级算术运算）和一组称为**VREGs**的向量寄存器组成，后者为VPU和MXU提供数据存储。

**VREGs：** 每个TPU v5p核心拥有64个32位VREGs（TPU v4为32个），这意味着每个核心的VREGs总存储容量约为 `64 * 8 * 128 * 4 = 256kB`（由于整个芯片有两个核心，总容量为此值的2倍）。TPU v5p每个周期可从VMEM加载3个寄存器，并写入1个寄存器到VMEM。

**VPU：** VPU是一个形状为`(8, 128)`的二维向量算术单元，其中128维称为**通道轴**，8维称为**子通道轴**。v5上每个（通道，子通道）对包含4个相互独立的浮点ALU。VPU在每个ALU中执行大多数算术指令（如vadd或向量加法）只需一个周期，延迟为2个周期。因此，例如在v5上，每个周期可以从VREGs中进行4对f32值的加法运算。一条典型的VPU指令可能如下所示：`{v2 = vadd.8x128.f32 v0, v1}`，其中v0和v1是输入VREGs，v2是输出VREG。

所有通道和子通道每个周期以纯SIMD方式执行相同的程序，但每个ALU可以执行不同的操作。因此，我们可以在一个周期内处理例如1个vadd和1个vsub指令，每条指令操作两个完整的VREGs并将结果写入第三个VREG。

**小测验 [计算VPU吞吐量]：** 使用以上信息，计算TPU v5p可以达到多少向量FLOPs/s。TPU v5p的时钟频率约为1.75GHz。

{% details 点击这里查看答案。 %}

*答案*：每个周期，每个核心可以在`8 * 128`个ALU上执行4条向量指令。这使得每个核心每个周期产生`8 * 128 * 4`次浮点运算（FLOPs），即`8 * 128 * 4 * 1.75e9 = 7e12 FLOPs/s`。请注意，这比MXU每核约`2e14` FLOPs/s的吞吐量小多少（大约小30倍）。

{% enddetails %}

**规约操作：** 通常，沿子通道维度的通信或规约操作比沿通道维度更容易。例如，VPU支持一种通道内混洗操作，可以在约一个周期内沿大小为8的轴滚动。这可用于高效地执行沿子通道维度的规约（只需进行4、2、1的混洗并执行3对元素级求和）。

跨通道规约则困难得多，需要一个称为XLU（"跨通道单元"）的独立硬件单元，该单元速度较慢且成本相当高。

**与GPU的对比：** 对于熟悉NVIDIA GPU的人来说，VPU中的每个ALU类似于一个CUDA核心，而单个VPU通道类似于一个"Warp调度器"，即通常由32个执行SIMD算术的CUDA核心组成的集合。通道内的规约操作相当容易，但如果我们需要跨通道，则需要至少经过VMEM/XLU/SMEM，这会慢得多。更多详情请参见[GPU章节](../gpus)。

### 标量核心

标量核心是TPU的控制单元。它负责获取和分发所有指令，执行从HBM到VMEM的数据传输，并且可编程处理标量元数据工作。由于标量核心是单线程的，这导致的一个副作用是TPU的每个核心每个周期只能创建一个DMA请求。

从这个角度看，单个标量核心控制着一个VPU（包含4096个ALU）、4个MXU、2个XLU以及多个DMA引擎。控制与计算资源高度不对称的特性是硬件效率的来源之一，但也限制了以任何有意义的方式进行数据相关向量化的能力。

### 附录 B：脉动阵列如何工作？

TPU MXU的核心是一个`128x128`的脉动阵列（TPU v6e为`256x256`）。在完全饱和的情况下，脉动阵列每8个时钟周期可执行一次`bf16[8,128] @ bf16[128,128] -> f32[8,128]`<d-footnote>如果你不熟悉这种表示法，它的意思是：将一个元素为bfloat16的`8x128`矩阵乘以一个元素为bfloat16的`128x128`矩阵，并将结果存储在一个元素为float32的`8x128`矩阵中。</d-footnote>矩阵乘法。

* 其核心是一个由`128x128`（`=16,384`）个ALU组成的二维网格，每个ALU都能执行一次乘加操作。
* 权重（**W**，`128x128`输入）从上方传入（称为右操作数或RHS），而输入数据（**X**，`8x128`输入）从左侧传入（称为左操作数或LHS）。

这是一个简化的动画，展示了将一组权重（蓝色）与一组激活值（绿色）相乘的过程。你会注意到权重（RHS）首先以对角线方式部分加载，然后激活值也以对角线方式输入。在下面的每一帧中，我们将所有重叠的绿色和蓝色单元相乘，将结果与从上方传入的任何残差求和，然后将结果向下传递一个单元。

{% include figure.liquid path="assets/img/systolic-array.gif" %}

这是一个更通用的动画版本，展示了计算结果如何被流出：

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

这是一个示意图，展示了如何通过多个RHS和LHS数组实现流水线操作：

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

在初始加载权重（RHS）和激活值（LHS）时，会有一个流水线气泡。初始气泡之后，可以在没有额外气泡的情况下加载新的输入和权重。

这是一个不太精确的动画，展示了一个`bf16[2, 3] x bf16[3, 3]`矩阵乘法。你可以将其想象为一个2x3权重矩阵与一个批量大小为1、尺寸为3的输入激活矩阵的乘法。与前面的幻灯片相比，这个动画是旋转过的，输入向右流出而不是向下，但你大致可以看到其结构。

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

我们可以有效地对此进行流水线化，以乘法较大的矩阵，同时不会产生过大的流水线气泡。话虽如此，重要的是我们的矩阵形状要大于MXU的边长维度（通常为128x128）。一些TPU（自TPU v3起）拥有多个MXU，TPU v3有2个，TPU v4/5有4个，因此我们需要确保分块维度大于128 * MXU的数量。[这里](https://www.youtube.com/watch?v=sJltBQ4MOHA)有一个很好的动画展示这一点。

Trillium（TPU v6e）拥有一个`256x256`的脉动阵列，这意味着它每周期可执行的浮点运算（FLOPs）数量是前代的4倍。这也意味着你的张量维度需要大一倍才能充分利用MXU。

[这篇博客文章](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)提供了另一个关于固定权重矩阵的脉动阵列乘法的精彩动画。