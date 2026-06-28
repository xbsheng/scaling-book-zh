---
layout: distill
title: "How to Think About GPUs"
description: "We love TPUs at Google, but GPUs are great too. This chapter takes a deep dive into the world of GPUs – how each chip works, how they're networked together, and what that means for LLMs, especially compared to TPUs. While there are a multitude of GPU architectures from NVIDIA, AMD, Intel, and others, here we will focus on NVIDIA GPUs. This section builds on <a href='https://jax-ml.github.io/scaling-book/tpus/'>Chapter 2</a> and <a href='https://jax-ml.github.io/scaling-book/training'>Chapter 5</a>, so you are encouraged to read them first."
date: 2025-08-18
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: "../conclusion"
previous_section_name: "Part 11: Conclusion"

next_section_url:
next_section_name: "The End"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin<sup>†</sup>
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: <sup>†</sup>Google DeepMind
  - name: Swapnil Patil<sup>†</sup>
    url: "https://www.linkedin.com/in/swapnil-patil-5b47a068"
  - name:  Adam Paszke<sup>†</sup>
    url: https://x.com/apaszke
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope
    affiliations:
      name: <sup>*</sup>MatX

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What Is a GPU?
  - subsections:
    - name: Memory
    - name: "Summary of GPU specs"
    - name: GPUs vs. TPUs at the chip level
    - name: "Quiz 1: GPU hardware"
  - name: Networking
  - subsections:
    - name: At the node level
    - name: "Quiz 2: GPU nodes"
    - name: Beyond the node level
    - name: "Quiz 3: Beyond the node level"
  - name: How Do Collectives Work on GPUs?
  - subsections:
    - name: Intra-node collectives
    - name: Cross-node collectives
    - name: "Quiz 4: Collectives"
  - name: "Rooflines for LLM Scaling on GPUs"
  - subsections:
    - name: "Data Parallelism"
    - name: "Tensor Parallelism"
    - name: "Expert Parallelism"
    - name: "Pipeline Parallelism"
    - name: "Examples"
    - name: "TLDR of LLM scaling on GPUs"
    - name: "Quiz 5: LLM rooflines"
  - name: "Acknowledgements and Further Reading"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: How does this change with GB200?"
    - name: "Appendix B: More networking details"

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

## What Is a GPU?
一个现代机器学习GPU（例如H100、B200）本质上就是一堆专门用于矩阵乘法（称为**流式多处理器**或**SM**）的计算核心，连接着一条高速内存（称为**HBM**）。这里有一张示意图：

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" link="true" caption="<b>图:</b> 展示H100或B200 GPU抽象布局的示意图。H100有132个SM，而B200有148个。我们比较宽泛地使用‘Warp调度器’这个术语来描述一组32个CUDA SIMD核心<i>以及</i>向其分配工作的调度器。注意这看起来多么像一个TPU！" %}

每个SM，就像TPU的张量核心一样，拥有一个专用的矩阵乘法核心（遗憾的是，也称为**张量核心**<d-footnote>GPU的张量核心是SM内的矩阵乘法子单元，而TPU的TensorCore是包含MXU、VPU和其他组件的总称。</d-footnote>），一个矢量算术单元（称为**Warp调度器**<d-footnote>NVIDIA没有为这个单元起一个好名字，所以我们只是在几个不好的选项中选了这个最好的。Warp调度器主要是向一组CUDA核心分配工作的单元，但我们在这里用它来描述控制单元及其所控制的核心集。</d-footnote>），以及一个快速的片上缓存（称为**SMEM**）。与最多只有2个独立“张量核心”的TPU不同，一个现代GPU拥有超过100个SM（H100上有132个）。这些SM中的每一个都比一个TPU张量核心弱得多，但整个系统更灵活。每个SM或多或少是完全独立的，因此一个GPU可以同时处理数百个独立任务。<d-footnote>尽管SM是独立的，但它们常常被迫为了达到峰值性能而进行协调，因为它们共享一个容量有限的L2缓存。</d-footnote>

让我们更详细地看一下一个H100的SM：

{% include figure.liquid path="assets/gpu/blackwell-sm.png" class="img-small" link="true" caption="<b>图:</b> H100 SM的示意图（<a href='https://wccftech.com/nvidia-hopper-gh100-gpu-official-5nm-process-worlds-fastest-hpc-chip-80-billion-transistors-hbm3-memory/'>来源</a>），展示了4个<i>子分区</i>，每个包含一个张量核心、Warp调度器、寄存器文件以及不同精度的CUDA核心集。底部附近的‘L1数据缓存’是256kB的SMEM单元。B200看起来类似，但增加了大量的张量内存（TMEM）来为庞大的张量核心提供数据。" %}

每个SM被分成4个相同的象限，NVIDIA称之为**SM子分区**，每个子分区包含一个张量核心、16k个32位寄存器，以及一个名为Warp调度器的SIMD/SIMT矢量算术单元，其通道（ALU）被NVIDIA称为**CUDA核心**。每个分区的核心组件可以说是张量核心，它执行矩阵乘法并构成其FLOPs/s的绝大部分，但它并不是唯一值得注意的组件。

* **CUDA核心:** 每个子分区包含一组称为CUDA核心的ALU，用于进行SIMD/SIMT矢量算术运算。每个ALU通常每个周期可以执行1个算术操作，例如`f32.add`。<d-footnote>较新的GPU支持FMA（融合乘加）指令，技术上每个周期执行两次浮点运算（FLOPs），NVIDIA毫不留情地利用这个事实来倍增其报告的规格。</d-footnote> 每个子分区包含32个fp32核心（以及较少数量的int32和fp64核心），它们都在每个周期执行相同的指令。就像TPU的VPU一样，CUDA核心负责ReLU、逐点矢量运算和归约（求和）。<d-footnote>历史上，在张量核心引入之前，CUDA核心是GPU的主要组件，用于渲染，包括光线-三角形求交和着色。在今天的游戏GPU上，它们仍然承担了大部分渲染工作，而TensorCore用于上采样（DLSS），这使得GPU可以以较低的分辨率（更少的像素=更少的工作量）进行渲染，然后使用机器学习进行上采样。</d-footnote>

* **张量核心（TC）:** 每个子分区都有自己的张量核心，这是一个类似于TPU MXU的专用矩阵乘法单元。张量核心代表了GPU FLOPs/s的绝大部分（例如，在H100上，我们有990 bf16 TC TFLOP/s，而CUDA核心只有66 TFLOPs/s）。
  * [990 bf16 TFLOPs/s](https://www.nvidia.com/en-us/data-center/h100/)，132个SM以1.76GHz运行，意味着每个H100 TC可以执行`7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOPs/周期，大约是一个8x8x8的矩阵乘法。<d-footnote>NVIDIA不分享太多TC硬件细节，所以这更像是一个猜测而不是确定的事实——当然，它没有说明TC是如何实现的。我们知道V100每个TC每周期可以执行256次浮点运算。A100可以执行512次，H100可以执行1024次，而B200的细节尚未公布，但似乎大约是2048 FLOPs/TC/周期，因为`2250e12 / (148 * 4 * 1.86e9)`大约是2048。一些更多细节<a href='https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727'>在此</a>得到确认。</d-footnote>
  * 与TPU一样，GPU可以以更高的吞吐量执行低精度矩阵乘法（例如H100的fp8 FLOPs/s是fp16的2倍）。低精度训练或推理可以显著更快。
  * 自Volta以来的每一代GPU都增加了TC的大小（[一篇关于此的好文章](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)）。到了B200，TC已经变得如此之大，以至于它无法将其输入放入SMEM中，因此B200引入了一种称为TMEM的新内存空间。<d-footnote>在Ampere中，张量核心可以从单个warp获取数据，而在Hopper中需要一个完整的SM（warp组），在Blackwell中则从2个SM获取数据。在Blackwell中，矩阵乘法也变得如此之大，以至于参数（特别是累加器）不再适合寄存器内存/SMEM，因此Blackwell添加了TMEM来解决这个问题。</d-footnote>

**CUDA核心比TPU的VPU更灵活：** GPU的CUDA核心（自V100起）使用所谓的SIMT（*单指令多线程*）编程模型，而TPU使用SIMD（*单指令多数据*）模型。与TPU VPU中的ALU一样，子分区内的CUDA核心必须在每个周期执行相同的操作（例如，如果一个核心在将两个浮点数相加，那么子分区中的所有其他CUDA核心也必须这样做）。然而，与VPU不同的是，每个CUDA核心（或CUDA编程模型中的“线程”）都有自己的指令指针，并且可以被_独立编程_。当同一个warp中的两个线程被指示执行不同的操作时，你实际上在_同时执行两个_操作，通过掩码屏蔽掉不需要执行分支操作的线程。

{% include figure.liquid path="assets/gpu/warp-divergence.png" class="img-fluid" caption="<b>图:</b> 线程集内warp divergence的示例（<a href='https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf'>来源</a>）。白色空间表示部分物理CUDA核心至少出现停滞。" %}

这使得在线程级别可以进行灵活的编程，但代价是如果warp太过频繁地分支，性能会默默地下降。线程在可以访问的内存方面也更加灵活；VPU只能操作连续的内存块，而CUDA核心可以访问共享寄存器中的单个浮点数并维护每线程的状态。

**CUDA核心调度也更灵活：** SM的运行有点像多线程CPU，因为它们可以同时“调度”多个程序（**warp**）（每个SM最多64个），但每个_Warp调度器_在每个时钟周期只执行单个程序。<d-footnote>调度在给定SM上的warp被称为“驻留”的。</d-footnote> Warp调度器会自动在活动的warp之间切换，以隐藏内存加载等I/O操作。相比之下，TPU通常是单线程的。

### 内存

除了计算单元，GPU还有一个内存层次结构，最大的是HBM（主GPU内存），然后是一系列较小的缓存（L2、L1/SMEM、TMEM、寄存器内存）。

* **寄存器:** 每个子分区都有自己的寄存器文件，在H100/B200上包含16,384个32位字（`4 * 16384 * 4 = 256kiB`每SM），可由CUDA核心访问。
  * 每个CUDA核心一次最多只能访问256个寄存器，因此尽管我们可以每个SM调度最多64个“驻留warp”，但如果每个线程使用256个寄存器，那么一次只能容纳8个（`256 * 1024 / (4 * 32 * 256)`）。

* **SMEM（L1缓存）:** 每个SM都有自己的256kB片上缓存，称为SMEM，它可以由程序员控制为“共享内存”，或被硬件用作片上缓存。SMEM用于存储激活值和TC矩阵乘法的输入。

* **L2缓存:** 所有SM共享<d-footnote>技术上，L2缓存被分成两半，因此在H100上，一半的SM可以访问25MB。两个半部分之间有连接，但带宽较低。</d-footnote>一个相对较大的约50MB的L2缓存，用于减少主内存访问。
  * 这与TPU的VMEM大小相似，但它**慢得多**，并且不受程序员控制。这导致了一点“超距作用”，程序员需要修改内存访问模式以确保L2缓存得到良好利用。<d-footnote>L2缓存在所有SM之间共享的事实，实际上迫使程序员以一种相当协调的方式运行SM，尽管原则上它们是独立的单元。</d-footnote>
  * NVIDIA不公布其芯片的L2带宽，但经过[测量](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth)大约为5.5TB/s。这大约是HBM带宽的1.6倍，但它是全双工的，所以有效的双向带宽更接近3倍。相比之下，TPU的VMEM大2倍*且*具有更高的带宽（约40TB/s）。

* **HBM:** 主GPU内存，用于存储模型权重、梯度、激活值等。
  * HBM容量从Volta的32GB大幅增加到Blackwell（B200）的192GB。
  * 从HBM到CUDA张量核心的带宽称为HBM带宽或内存带宽，在H100上约为3.35TB/s，在B200上约为9TB/s。

### GPU规格摘要

以下是近期型号的GPU规格摘要。给定GPU的变体在SM数量、时钟速度和FLOPs方面略有不同。这里是内存容量数据：

|  GPU  | 代次 |   时钟速度   | SMs/芯片 | SMEM容量/SM | L2容量/芯片 | HBM容量/芯片 |
| :---: | :--------: | :-------------: | :------: | :--------------: | :--------------: | :---------------: |
| V100  |   Volta    | 1.25GHz/1.38GHz |    80    |       96kB       |       6MB        |       32GB        |
| A100  |   Ampere   | 1.10GHz/1.41GHz |   108    |      192kB       |       40MB       |       80GB        |
| H100  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       80GB        |
| H200  |   Hopper   | 1.59GHz/1.98GHz |   132    |      256kB       |       50MB       |       141GB       |
| B200  | Blackwell  |        ?        |   148    |      256kB       |      126MB       |       192GB       |

所有代次每SM都有256kB的寄存器内存。Blackwell每SM还增加了256kB的TMEM。以下是每块芯片的FLOPs和带宽数字：

|  GPU  | 代次 | HBM 带宽/芯片 | FLOPs/s/芯片 (bf16/fp16) | FLOPs/s/芯片 (fp8/int8) | FLOPs/s/芯片 (fp4) |
| :---: | :--------: | :---------: | :----------------------: | :---------------------: | :----------------: |
| V100  |   Volta    |   9.0e11    |            —             |            —            |         —          |
| A100  |   Ampere   |   2.0e12    |          3.1e14          |         6.2e14          |         —          |
| H100  |   Hopper   |   3.4e12    |          9.9e14          |         2.0e15          |         —          |
| H200  |   Hopper   |   4.8e12    |          9.9e14          |         2.0e15          |         —          |
| B200  | Blackwell  |   8.0e12    |          2.3e15          |         4.5e15          |       9.0e15       |

我们排除了B100，因为它没有大规模生产。<d-footnote>虽然NVIDIA生产了B100一代，但它们只被短暂地销售和生产，据称是由于设计缺陷，使其无法接近其声称的规格运行。它们在热量和功耗问题的影响下难以达到峰值FLOPs。</d-footnote> 一些规格取决于GPU的确切版本，因为NVIDIA GPU不像TPU那样标准化。

这里有一张有用的备忘单，比较了GPU和TPU的组件：

|              GPU              |     TPU     |              功能              |
| :---------------------------: | :---------: | :-----------------------------------: |
| 流式多处理器（SM） | 张量核心 | 包含其他单元的核心“单元” |
|        Warp调度器         |     VPU     |      SIMD矢量算术单元      |
|           CUDA核心           |   VPU ALU   |               SIMD ALU                |
|        SMEM（L1缓存）        |    VMEM     |       快速片上缓存内存       |
|          张量核心          |     MXU     |      �
## Networking
网络是GPU和TPU差异最大的领域之一。如前所述，TPU以2D或3D环面拓扑连接，每个TPU仅与相邻TPU相连。这意味着两个TPU之间的消息传递必须经过中间的每一个TPU，并迫使我们在网格上只能使用统一的通信模式。虽然这在某些方面带来不便，但同时也意味着每个TPU的链路数量恒定，我们可以扩展到任意规模的TPU "pod"而不损失带宽。

而GPU则采用更传统的分层树状交换网络。由8个GPU组成的集合称为**节点**（GB200中最多可达72个<d-footnote>“节点”一词有重载含义，可指代两种概念：NVLink域（即通过NVLink互连完全连接的GPU集合），或连接到单个CPU主机的GPU集合。在B200之前，这两者通常一致，但在GB200 NVL72中，我们拥有72个GPU的NVLink域，但仍只有8个GPU连接到每个主机。此处我们使用“节点”指代NVLink域，但这存在争议。</d-footnote>），这些GPU使用名为NVLink的高带宽互连在1跳内相互连接。这些节点通过附加到每个GPU的NIC，使用较低带宽的InfiniBand（IB）或以太网连接成更大的单元（称为**SU**或可扩展单元）。这些单元进而可以通过更高层的交换机连接成任意规模的单元。

{% include figure.liquid path="assets/gpu/superpod-diagram.png" class="img-fluid" caption="<b>图：</b>典型H100网络示意图。一组8个GPU通过NVSwitch（也称NVLink交换机）连接成节点或NVLink域，这些节点通过交换式InfiniBand结构相互连接。H100在NVLink域内每GPU出口带宽约450GB/s，每个节点到IB网络的出口带宽为400GB/s。" %}

### 节点层面

GPU节点是一个小型单元，通常包含8个GPU（GB200中可达72个），通过全对全、全带宽、低延迟的NVLink互连。<d-footnote>NVLink被描述为一种增强版PCIe连接，具有低延迟和低协议开销，但未为可扩展性/容错性设计；而InfiniBand更类似以太网，专为大型有损网络设计。</d-footnote>每个节点包含多个高带宽NVSwitch，用于在所有本地GPU间交换数据包。实际的节点级拓扑随时间变化很大，包括每节点交换机数量，但对于H100，我们有每节点4个NVSwitch，GPU以`5 + 4 + 4 + 5`链路模式连接，如图所示：

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>图：</b>自Pascal（P100）起的节点（即NVLink域）示意图。自Volta（V100）起，节点内通过一组交换机实现全对全连接。H100节点有4个NVSwitch，通过25GB/s链路连接所有8个GPU。" %}

对于Hopper架构（NVLink 4.0），每条NVLink链路具有25GB/s的全双工<d-footnote>此处全双工指每个方向25GB/s，且双向独立。链路总传输能力为50GB/s，但每个方向最多25GB/s。</d-footnote>带宽（B200为50GB/s），使每个GPU到网络具有`18 * 25=450GB/s`的全双工带宽。大型NVSwitch最多拥有64个NVLink端口，这意味着配备4个交换机的8xH100节点可处理高达`64 * 25e9 * 4=6.4TB/s`的带宽。以下是各GPU世代这些数值的变化概览：

| NVLink代次 | NVSwitch代次 | GPU架构 | NVLink带宽 (GB/s，全双工) | 每GPU NVLink端口数 | 节点内GPU间带宽 (GB/s 全双工) | 节点规模 (NVLink域) | 每节点NVSwitch数 |
| :--------: | :----------: | :-----: | :----------------------: | :----------------: | :-------------------------: | :-----------------: | :-------------: |
|   **3.0**  |    **2.0**   | Ampere  |            25            |         12         |             300             |          8          |        6        |
|   **4.0**  |    **3.0**   | Hopper  |            25            |         18         |             450             |          8          |        4        |
|   **5.0**  |    **4.0**   |Blackwell|            50            |         18         |             900             |        8/72         |      2/18       |

Blackwell（B200）拥有8 GPU节点。GB200NVL72支持更大的72 GPU NVLink域。我们将展示8 GPU和72 GPU系统的详细信息。

### 测验2：GPU节点

以下是更多关于网络的问答题。我认为实际推导这些题目特别有用，因为它们能让你深入理解实际通信模式。

**问题1 [H100节点总带宽]：** 在配备4个交换机的8xH100节点中，每节点总带宽是多少？*提示：* 同时考虑NVLink和NVSwitch带宽。

{% details 点击查看答案。 %}

**答案：** 我们有4个Gen4 NVSwitch，每个具有`64 * 25e9=1.6TB/s`单向带宽。交换机层面总带宽为`4 * 1.6e12=6.4e12`。但请注意，每个GPU最多处理450GB/s单向带宽，因此实际最大带宽为`450e9 * 8 = 3.6TB/s`。由于此值更小，故峰值带宽为3.6TB/s。

{% enddetails %}

**问题2 [对分带宽]：** 对分带宽定义为网络任意均分分区间的最小可用带宽。换言之，若将网络分成两个相等部分，两部分间的带宽是多少？你能计算8x H100节点的对分带宽吗？*提示：* 对分带宽通常包含双向流量。

{% details 点击查看答案。 %}

**答案：** 任意均分分区每部分有4个GPU，每个GPU可向另一部分出口`4 * 450GB/s`。计入双向流量，跨分区总流量为`8 * 450GB/s`，即3.6TB/s对分带宽。这与NVIDIA报告一致，例如[此处](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)。

{% enddetails %}

**问题3 [AllGather成本]：** 给定B字节数组，在8xH100节点上执行（吞吐量受限的）AllGather需要多长时间？请计算bf16[D<sub>X</sub>, F]格式（其中`D=4096`, `F=65,536`）的时间。*回答前建议先阅读TPU集合操作[章节](https://jax-ml.github.io/scaling-book/sharding/)。在此思考，后续将深入讨论集合操作。*

{% details 点击查看答案。 %}

**答案：** 每个GPU出口带宽450GB/s，每个GPU持有$B / N$字节（`N=8`为节点规模）。可想象每个节点依次向其他$N - 1$个节点发送数据，共需(N-1)轮，每轮通信时间$T_\text{comms} = (B / (N * W_\text{单向}))$，总时间$T_\text{comms} = (N - 1) * B / (N * W_\text{单向})$。约等于$B / (N * W_\text{单向})$或$B / \text{3.6e12}$（对分带宽）。

对于给定数组，`B=4096 * 65536 * 2=512MB`，总时间`536e6 * (8 - 1) / 3.6e12 = 1.04ms`。由于可能受限于延迟，实际时间可能更长（实测约1.5ms）。

{% enddetails %}
## Beyond the node level
在节点级别之上，GPU 网络的拓扑标准化程度较低。NVIDIA 发布了一份[参考 DGX SuperPod 架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html)，该架构使用 InfiniBand 连接比单节点更多的 GPU，但客户和数据中心提供商可以根据自身需求自由定制此架构。<d-footnote>例如，Meta 在一个数据中心网络上训练了 LLaMA-3，该网络与此描述大不相同，它使用了以太网、一个三层交换结构，以及顶层的一个过载交换机。</d-footnote>

以下是一个参考 1024 GPU H100 系统的示意图，其中底排的每个方框代表一个单节点，包含 8 块 H100 GPU、8 个 400Gbps CX7 NIC（每个 GPU 对应一个）和 4 个 NVSwitch。

{% include figure.liquid path="assets/gpu/h100-superpod.png" class="img-fluid" caption="<b>图：</b> 参考 1024 H100 DGX SuperPod 架构示意图，包含 128 个节点（有时是 127 个），每个节点有 8 块 H100 GPU，连接到一个 InfiniBand 横向扩展网络。每组 32 个节点（256 块 GPU）被称为“可扩展单元”或 SU。叶脊 IB 交换机提供了足够的带宽，确保节点间的全等分带宽。" %}

**可扩展单元：** 每组 32 个节点被称为一个“可扩展单元”（或 SU），由一组 8 台叶 InfiniBand 交换机管理。这个 SU 有 256 块 GPU，每个节点有 4 个 NVSwitch，并配有 8 台 InfiniBand 叶交换机。图中显示的所有布线都是 InfiniBand NDR（50GB/s 全双工），并使用 64 端口的 NDR IB 交换机（每端口也是 50GB/s）。*请注意，IB 交换机的带宽是 NVSwitch 的 2 倍（64 个端口，每个端口 400 Gbps 链路）。*

**SuperPod：** 整个 SuperPod 随后将 4 个这样的 SU 与 16 台顶层“脊”IB 交换机连接起来，构成一个拥有 1024 块 GPU 的系统，包含 512 个节点级 NVSwitch、32 台叶 IB 交换机和 16 台脊 IB 交换机，总计 512 + 32 + 16 = 560 台交换机。叶交换机以每组 32 个节点的方式连接到节点，因此每组 256 块 GPU 有 8 台叶交换机。所有叶交换机都连接到所有脊交换机。

**我们有多少带宽？** InfiniBand 网络（称为“横向扩展网络”）的整体拓扑是一个**胖树**，其电缆和交换机保证了在节点级别之上的全等分带宽（此处为 400GB/s）。这意味着如果我们把节点分成两半，每个节点可以同时向另一半分区中的节点发送 400GB/s 的数据。更重要的是，这意味着我们应该在横向扩展网络中获得大致恒定的 AllReduce 带宽！虽然可能并非这样实现，但你可以想象在横向扩展网络中对任意数量的节点进行环归约，因为你可以构建一个包含所有节点的环。

| 级别 | GPU 数量 | 每单元交换机数量 | 交换机类型 | 每单元带宽 (TB/s, 全双工) | GPU 间带宽 (GB/s, 全双工) | 胖树带宽 (GB/s, 全双工) |
| :---: | :------------: | :-------------------------: | :---------: | :------------------------------------------: | :--------------------------------------: | :---: |
| 节点 |       8        |              4              |     NVL     |                     3.6                      |                   450                    | 450
| 叶   |      256       |              8              |     IB      |                     12.8                     |                    50                    | 400 |
| 脊   |      1024      |             16              |     IB      |                     51.2                     |                    50                    | 400 |

相比之下，一个 TPU v5p 每链路大约有 90GB/s 的出站带宽，或者在 3D 环面的所有轴上总计 540GB/s 的出站带宽。这不是点对点的，因此只能用于受限的、统一的通信模式，但它仍然为我们提供了更高的 TPU 间带宽，并且可以扩展到任意大的拓扑（至少达到 8960 个 TPU）。

理论上，GPU 交换结构可以通过添加额外的交换机或间接层来扩展到任意规模，代价是增加延迟和昂贵的网络交换机。

<p markdown=1 class="takeaway">**要点**：在 H100 节点内部，我们从每个 GPU 获得 450GB/s 的全胖树带宽，而在节点之外，节点间带宽降至 400GB/s。这对于通信原语来说至关重要。</p>

**GB200 NVL72s：** NVIDIA 最近开始生产新的 GB200 NVL72 GPU 集群，这些集群将 72 块 GPU 集成在单个 NVLink 域中，具有完整的 900GB/s 的 GPU 间带宽。这些域随后可以链接成更大的 SuperPod，并配备相应增加的（9 倍）IB 胖树带宽。以下是该拓扑的示意图：

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b> 一个包含 576 块 GPU 的 GB200 DGX SuperPod 示意图。底层的每个机架包含 72 块 GB200 GPU。" %}

计算单个节点的出站带宽（上图中的橙色线），我们有 `4 * 18 * 400 / 8 = 3.6TB/s` 的带宽到达叶级别，这比 H100 多了 9 倍（正如节点包含多 9 倍的 GPU 一样）。这意味着关键的节点出站带宽要高得多，*非常高*，而我们的跨节点集合带宽实际上可能*低于*节点内部带宽。
更多讨论见[附录 A](#appendix-a-how-does-this-change-with-gb200)。

|  节点类型   | 每节点 GPU 数量 | GPU 出站带宽 | 节点出站带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

<p markdown=1 class="takeaway">**要点**：GB200 NVL72 SuperPod 大幅增加了节点尺寸和特定节点的出站带宽，这显著改变了我们的性能天花板。</p>

### 测验 3：节点级别之上

**问题 1 [胖树拓扑]：** 使用上面的 DGX H100 示意图，计算整个 1024 GPU Pod 在节点级别的等分带宽。证明链路带宽的选择确保了全等分带宽。*提示：确保同时计算链路带宽和交换机带宽。*

{% details 点击此处查看答案。 %}

**答案：** 让我们逐个组件进行计算：

* 首先，每个节点有 8 条 400Gbps NDR IB 电缆连接到叶交换机，为每个节点提供 `8 * 400 / 8 = 400 GB/s` 到叶交换机的带宽。我们有 8 台叶交换机，每台带宽为 3.2TB/s（64 条 400 GBps 链路），但我们只能使用 64 个端口中的 32 个从 SU 接收数据，所以对于 32 个节点来说是 `32 * 400 / 8 = 12.8TB/s`，同样正好是 400GB/s 每个节点。
* 然后在脊级别，我们有 `8 * 16 * 2` 条 400Gbps NDR IB 电缆将每个 SU 连接到脊，为每个 SU 提供 `8 * 16 * 2 * 400 / 8 = 12.8 TB/s` 到叶的带宽。这同样是每个节点 400GB/s。我们有 16 台脊交换机，每台带宽为 3.2TB/s，总共 `16 * 3.2 = 51.2 TB/s`，对于 128 个节点来说，同样是 400GB/s。

因此，如果我们以任何方式将节点对半分开，节点间的每 GPU 带宽将是 400GB/s。每个组件都恰好具有确保胖树所需的带宽。

{% enddetails %}

**问题 2 [扩展到更大的 DGX Pod]：** 假设我们想在 2048 个 GPU 上训练，而不是 1024 个。修改上述 DGX 拓扑以处理此需求的最简单/最佳方式是什么？4096 个呢？*提示：没有唯一正确的答案，但要尽量降低成本。请记住链路容量。[本文档](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)可能有所帮助。*

{% details 点击此处查看答案。 %}

**答案：** 一种选择是保持 SU 结构不变（32 个节点由 8 台交换机管理），只需增加更多 SU 和更多顶层交换机。我们需要多 2 倍的脊交换机，所以我们会有 8 个 SU，配备 32 台脊交换机，提供足够的带宽。

这样做的一个问题是，我们每台叶交换机只有 64 个端口，在上面的示意图中我们已经使用了全部。但相反，我们可以轻松地对每台脊交换机只用 1 条 400 Gbps NDR 电缆，而不是 2 条，这提供了相同的总带宽，但节省了一些端口。

对于 4096 个 GPU，我们的端口实际上会不够用，因此我们需要添加另一个间接层，也就是说，在层次结构中增加一级。NVIDIA 称这些为“核心交换机”，并构建了一个包含 128 台脊交换机和 64 台核心交换机的 4096 GPU 集群。你可以计算一下，证明这提供了足够的带宽。

{% enddetails %}
## How Do Collectives Work on GPUs?
GPU可以执行与TPU完全相同的集合操作：ReduceScatter、AllGather、AllReduce和AllToAll。与TPU不同，这些操作的工作方式取决于它们是在节点级别（通过NVLink）还是在更高级别（通过InfiniBand）执行。这些集合操作由NVIDIA在[NVSHMEM](https://developer.nvidia.com/nvshmem)和[NCCL](https://developer.nvidia.com/nccl)（发音为"nickel"）库中实现。NCCL已在此处开源[链接](https://github.com/NVIDIA/nccl)。尽管NCCL根据延迟要求和拓扑结构使用多种实现方式（[详情](https://github.com/NVIDIA/nccl/issues/1415#issuecomment-2310650081)），但从现在开始，我们将讨论在交换树状拓扑上理论最优的模型。

### 节点内集合操作

**AllGather或ReduceScatter：** 对于节点级别的AllGather或ReduceScatter，你可以像TPU那样围绕环进行操作，在每一跳使用GPU间全带宽。将GPU任意排序，并使用GPU间全带宽沿环发送数组的一部分。<d-footnote>你也可以认为每个GPU向其他$N - 1$个GPU各发送其大小为$\text{bytes} / N$的块，总共传输$(N - 1) * N * bytes / N$字节数据，这给出了相同的结果。</d-footnote>每一跳的成本是$T_\text{hop} = \text{bytes} / (N * \text{GPU出口带宽})$，因此总成本为

$$T_\text{AG或RS通信} = \frac{\text{字节数} \cdot (N - 1)}{N \cdot \text{GPU出口带宽}} \rightarrow \frac{\text{字节数}}{\text{GPU出口带宽}}$$

你会注意到这与TPU完全相同。对于AllReduce，你可以像往常一样组合RS + AG，但成本是两倍。

{% include figure.liquid path="assets/gpu/all-gather.gif" class="img-fluid" caption="<b>图：</b> 带宽最优的1D环AllGather算法。对于B字节数据，通过顶层交换机发送B / X字节X - 1次。" %}

如果你关心延迟（例如，如果你的数组非常小），你可以进行树状归约，即先在每2个GPU间AllReduce，然后在每4个，然后在每8个，总共$\log(N)$跳而不是$N - 1$跳，尽管总成本仍然相同。

<p markdown=1 class="takeaway">**要点：** 在单个节点内对B字节数组进行AllGather或ReduceScatter的成本大约是$T_\text{通信} = B * (8 - 1) / (8 * W_\text{GPU出口带宽}) \approx B / W_\text{GPU出口带宽}$。在H100上理论值约为$B / \text{450e9}$，在B200上约为$B / \text{900e9}$。除非启用网络内归约，否则AllReduce的成本是此值的2倍。</p>

<b markdown=1 style="color: #57cf57;">随堂测验 1 [AllGather时间]：</b> 使用一个具有450 GB/s全双工带宽的8xH100节点，AllGather(bf16[B<sub>X</sub>, F])需要多长时间？设$B=1024$，$F=16,384$。

{% details 点击此处查看答案。 %}

**答案：** 我们总共有$2 \cdot B \cdot F$字节，单向带宽为450e9。这将需要大约$T_\text{通信} = (2 \cdot B \cdot F) / \text{450e9}$，或者更精确地说是$(2 \cdot B \cdot F \cdot (8 - 1)) / (8 \cdot \text{450e9})$。使用给定的值，我们得到大约$(2 \cdot 1024 \cdot 16384) / \text{450e9} = \text{75us}$，更精确地说是$\text{65us}$。

{% enddetails %}

**AllToAll：** 节点内的GPU具有全连接性，这使得AllToAll操作相当容易。每个GPU只需直接发送到目标节点。在节点内，对于B字节数据，每个GPU有$B / N$字节，并向$N - 1$个目标节点各发送$(B / N^2)$字节，总共

$$T_\text{AllToAll通信} = \frac{B \cdot (N - 1)}{W \cdot N^2} \approx \frac{B}{W \cdot N}$$

将其与TPU比较，TPU的成本是$B / (4W)$。因此，在单个节点内，我们得到了2倍的理论运行时加速（$B / 4W$ 对 $B / 8W$）。

对于混合专家（MoE, Mixture of Expert）模型，我们经常想要执行*稀疏或非均匀的AllToAll*，即保证输出维度上$N$个分片中最多有$k$个非零，也就是说$T_\text{AllToAll} \rightarrow K[B, N]$，其中每个轴上$N$个条目中最多有$k$个非零。其成本按$k/N$比例减少，总共约为$\min(k/N, 1) \cdot B / (W \cdot N)$。对于MoE，我们通常独立随机选择非零值，因此有少于$k$个非零的可能性，给出的近似值为
$(N-1)/N \cdot \min(k/N, 1) \cdot B / (W \cdot N)$。<d-footnote>实际成本实际上是$$(1 - \left(\frac{Z - 1}{Z}\right)^K) \cdot \frac{Z - 1}{Z}$$，即$K$次掷骰子中不同结果的期望数量，但它与给出的近似值非常接近。更多细节请参见附录。</d-footnote>

<b markdown=1 style="color: #c55404ff;">随堂测验 2 [AllToAll时间]：</b> 使用一个具有450 GB/s单向带宽的8xH100节点，AllToAll<sub>X->N</sub>(bf16[B<sub>X</sub>, N])需要多长时间？如果我们知道只有8个条目中的4个是非零的呢？

{% details 点击此处查看答案。 %}

**答案：** 根据上文，在稠密情况下，成本是$B \cdot (N-1) / (W \cdot N^2)$，即$B / (W \cdot N)$。如果我们知道只有$\frac{1}{2}$的条目是非填充的，我们可以发送$B \cdot k/N / (W \cdot N) = B / (2 \cdot W \cdot N)$，大约是总成本的一半。

{% enddetails %}

<p markdown=1 class="takeaway">**要点：** 在单个节点内的GPU上，对$B$字节数组执行AllToAll的成本大约是$T_\text{通信} = (B \cdot (8 - 1)) / (8^2 \cdot W_\text{GPU出口带宽}) \approx B / (8 \cdot W_\text{GPU出口带宽})$。对于非均匀（top-$k$）AllToAll，此成本进一步降低到$(B \cdot k) / (64 \cdot W_\text{GPU出口带宽})$。</p>

**实测数据：** 这里是8xH100节点上AllReduce带宽的实测数据。Algo BW是测量带宽（字节数 / 运行时间），Bus BW计算为$2 \cdot W \cdot (8 - 1) / 8$，理论上是实际链路带宽的度量。你会注意到我们确实达到了接近370GB/s，低于450GB/s但相当接近，尽管每设备仅约10GB。这意味着虽然这些估计在理论上是正确的，但需要非常大的消息才能达到。

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b> 8xH100节点在禁用SHARP时的AllReduce吞吐量。蓝色曲线是经验链路带宽，根据经验测量计算为 $2 * \text{字节数} * (N - 1) / (N * \text{运行时间})$。请注意，即使使用巨大的10GB数组，我们也未能特别接近标称的450GB/s带宽。" %}

这是一个实际问题，因为它有意义地复杂化了我们能做出的任何理论断言，因为例如，即使是对一个合理大小的数组进行AllReduce，如LLaMA-3 70B的MLP（大小为`bf16[8192, 28672]`，或者采用8路模型分片时为`bf16[8192, 3584] = 58MB`），也只能达到约150GB/s，而峰值是450GB/s。相比之下，TPU在更低的消息大小下就能达到峰值带宽（见附录B）。

<p markdown=1 class="takeaway">**要点：** 尽管NVIDIA声称H100 NVLink的带宽约为450GB/s，但在实践中很难超过370 GB/s，因此请相应地调整上述估计。</p>

**网络内归约：** 从Hopper架构开始，NVIDIA交换机支持["SHARP"（可扩展分层聚合与归约协议）](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/)，这允许"网络内归约"。这意味着*网络交换机本身*可以执行归约操作，并将结果多路复用或"多播"到多个目标GPU：

{% include figure.liquid path="assets/gpu/sharp-algorithm.png" class="img-fluid" caption="<b>图：</b> 一个没有SHARP的AllReduce有2倍的理论成本，因为它必须两次通过每个GPU。在实践中，加速仅为约30%（基于NCCL 2.27.5）。" %}

理论上，这几乎将AllReduce的成本减半，因为这意味着每个GPU可以将其数据发送到顶层交换机，由交换机本身执行归约并将结果广播到每个GPU，而无需让每个GPU出口两次，同时也降低了网络延迟。

$$T_\text{SHARP AR通信} = \frac{\text{字节数}}{\text{GPU出口带宽}}$$

请注意这是精确的，没有$1/N$的误差因子，因为每个GPU首先出口$B \cdot (N - 1) / N$字节，然后接收其本地分片部分归约的版本（入口$B/N$字节），完成归约，然后再次出口$B/N$字节，最后入口完全归约的结果（入口$B \cdot (N - 1) / N$字节），总共入口恰好$B$字节。

然而，在实践中，我们看到启用SHARP后带宽增加约30%，而预测的是75%。这只能使有效集合带宽提升到约480GB/s，远非2倍。

{% include figure.liquid path="assets/gpu/sharp-all-reduce-cost.png" class="img-fluid" caption="<b>图：</b> 在节点内启用和未启用NVIDIA SHARP时的AllReduce算法带宽实测。峰值吞吐量提升约30%，尽管算法上应该能达到接近75%的增益。" %}

<p markdown=1 class="takeaway">**要点：** 理论上，NVIDIA SHARP（在大多数NVIDIA交换机上可用）应将对$B$字节的AllReduce成本从大约$2 * B / W$降低到$B / W$。然而，在实践中我们只看到带宽大约30%的提升。由于纯AllReduce在LLM中相当罕见，这用处不大。</p>

### 跨节点集合操作

当我们超出节点级别时，成本会更加微妙。当在树状结构上执行归约时，你可以自底向上考虑归约，首先在节点内，然后在叶子层，最后在主干层，每一层使用常规算法。特别是对于AllReduce，你可以看到这允许我们通信更少的数据，因为我们在节点级别AllReduce之后，只需向叶子节点出口$B$字节，而不是$B * N$。

**成本有多高？** 作为一级近似，因为我们拥有全对分带宽，所以AllGather或ReduceScatter的成本大致是缓冲区字节大小除以节点出口带宽（H100上为400GB/s），*与树状归约的任何细节无关*。

$$T_\text{AG或RS通信} = \frac{\text{字节数}}{W_\text{节点出口}} \underset{H100}{=} \frac{\text{字节数}}{\text{400e9}}$$

其中$W_\text{节点}$出口对于上述H100网络（每个节点有8x400Gbps IB链路出口）通常为400GB/s。理解这一点的最清晰方式是，想象在集群中的*每个节点*上执行环形归约。由于胖树拓扑结构，我们总能在任意两个节点之间构造一个具有$W_\text{节点}$出口带宽的环，并执行常规归约。节点级别的归约（几乎）永远不会成为瓶颈，因为它具有更高的总体带宽和更好的延迟，尽管通常成本为

$$T_\text{总} = \max(T_\text{节点内通信}, T_\text{扩展网络通信}) = \max\left[\frac{\text{字节数}}{W_\text{GPU出口}}, \frac{\text{字节数}}{W_\text{节点出口}}\right]$$

{% details 你可以在此处查看更精确的推导。 %}

我们可以更精确地指出，我们实际上是在网络的每一层执行环形归约，这些操作大部分可以重叠，所以我们有：

$$T_\text{AG或RS通信} = \text{字节数} \cdot \max_{\text{深度 i}}\left[\frac{D_i - 1}{D_i \cdot W_\text{链路 i}}\right]$$

其中$D_i$是深度$i$处的度（深度$i$处子节点的数量），$W_\text{链路 i}$是连接每个子节点到节点$i$的链路带宽。

利用这个公式，我们可以计算给定拓扑的可用AllGather/AllReduce带宽为$\min_{\text{深度 i}}(D_i * W_\text{链路 i} / (D_i - 1))$。在上述情况下，我们有：

* **节点：** $D_\text{节点}$ = 8，因为节点内有8个GPU，$W_\text{链路 i}$ = 450GB/s。因此，我们有AG带宽 `450e9 * 8 / (8 - 1) = 514GB/s`。
* **叶子：** $D_\text{叶子}$ = 32，因为一个SU（SuperUnit）中有32个节点，$W_\text{链路 i}$ = 400GB/s（8x400Gbps IB链路）。因此，我们的带宽是 `400e9 * 32 / (32 - 1) = 413GB/s`。
* **主干：** $D_\text{主干}$ = 4，因为我们有4个SU，$W_\text{链路 i}$ = 12.8TB/s（来自上面的 `8 * 16 * 2 * 400Gbps` 链路）。我们的带宽是 `12.8e12 * 4 / (4 - 1) = 17.1TB/s`。

因此，我们的总AG或RS带宽是叶子层的 `min(514GB/s, 413GB/s, 17.1TB/s) = 413GB/s`，所以在实践中 $T_\text{AG或RS通信} = B / \text{413GB/s}$，即即使在最高级别，我们也大约有413GB/s的AllReduce带宽。对于启用SHARP的AllReduce，它将略低于此（约400GB/s），因为我们没有$(N - 1) / N$这个因子。尽管如此，450GB/s和400GB/s足够接近，可以作为近似值使用。

{% enddetails %}

**其他集合操作：** AllReduce的成本仍然是上述的2倍，
## Rooflines for LLM Scaling on GPUs
现在让我们来看看这一切的目标：理解GPU上LLM扩展的屋顶线模型（roofline）。这部分内容将作为TPU训练章节[此处](../training)的补充。与那里相同，我们的目标是分析不同并行策略下的总计算时间 $T_\text{math}$ 和通信时间 $T_\text{comms}$，并理解在何时会出现 $T_\text{comms} > T_\text{math}$。和之前一样，我们仅考虑包含以下操作的MLP（多层感知机）模块：

$$\text{MLP}(x) \equiv x[B, D] *_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D]$$

其中 $B$ 是**全局批次大小（以token为单位）**（即 $B = \text{批次大小} \cdot \text{序列长度}$）。

这里我们将重现上文中的表格，展示GPU和节点级别的有效带宽：

| 节点类型 | 每节点GPU数 | GPU出口带宽 | 节点出口带宽 |
| :---------: | :-----------: | :------------------: | :-------------------: |
|    H100     |       8       |        450e9         |         400e9         |
|    B200     |       8       |        900e9         |         400e9         |
| GB200 NVL72 |      72       |        900e9         |        3600e9         |

**注意：** GPU和节点出口带宽共同决定了我们LLM的屋顶线。我们将使用术语 $W_\text{collective}$ 来描述GPU或节点带宽，具体取决于我们是在节点内部还是跨节点进行操作。

我们将像分析TPU那样，查看**数据并行（data parallelism）、张量并行（tensor parallelism）、流水线并行（pipeline parallelism）、专家并行（expert parallelism）**及其组合的计算通信屋顶线。在本节的其余部分，我们将重点分析H100在特定计算场景下的屋顶线。GB200-NVL72具有相同的整体屋顶线，但由于节点出口带宽更大，有时瓶颈可能出现在节点级别。

### 数据并行

如前所述，DP（数据并行）和ZeRO分片在反向传播中涉及权重AllReduce或ReduceScatter + AllGather操作。由于这两者开销相同，为了在纯数据并行或FSDP（*不启用网络内归约*）中成为计算受限，对于每一层，在反向传播中，拥有一个大小为X的维度时，我们有：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF}{W_\text{collective}}$$

因此，对于 $T_\text{math} > T_\text{comms}$，我们需要 $B / (XC) > 1 / W_\text{collective}$，或

$$\frac{B}{X} > \frac{C}{W_\text{collective}}$$

其中 $W_\text{collective}$ 是GPU或节点级出口带宽，具体取决于我们是在节点内还是跨节点进行分片。因此：

* **在节点内部**，我们只需要每个GPU的**token**批次大小 > $\text{990e12} / \text{450e9} = 2200$。
* **在SU（超级单元）内部或spine（骨干）级别**，BS > $\text{990e12} / \text{400e9} = 2475$。

这比TPU上的数值要高不少，TPU上三个轴向的数字是850。例如，在16000个H100上训练的LLaMA-3，其批次大小至少需要40M token（作为参考，他们使用了16M）。在2048个H800 GPU上训练的DeepSeek v3，其带宽较低为300GB/s（H100上为450GB/s），则需要 $\text{990e12} / \text{300e9} = 3300$ 个token/GPU，即大约6.7M（实践中，他们使用了4M）。

启用网络内归约并使用纯数据并行时，理论上我们拥有2倍的AllReduce带宽，这将使上述两个数字减半。然而，实践中收益接近30%，这实际上仅弥补了我们通常难以达到报告数字的事实。此外，由于纯数据并行很少使用，这在实践中基本上无关紧要。

**混合专家（MoE）模型：** 对于拥有E个专家、每个token激活k个专家的混合专家模型，上述公式变为：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot EDF}{W_\text{collective}}$$

这使得每个GPU的token批次大小增加了 $E/k$ 倍，即

$$\frac{B}{X} > \frac{E}{k} \frac{C}{W_\text{collective}}$$

例如，新的OpenAI开源模型 $k=4$ 且 $E=128$，跨节点时此数值增加到 `32 * 2475 = 79,200`，这是一个高得离谱的数字。

**当X很小时会发生什么？** 当我们只进行例如2节点数据并行时，我们受益于 $(X - 1) / X$ 的缩放比例，这给了我们：

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{N * C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot 2 \cdot DF \cdot (X-1)}{X \cdot W_\text{collective}}$$

其中X是节点数，$N = 8 \cdot X$。那么对于一个稠密模型，我们有 $B / N > \alpha \cdot (X - 1) / X$，或例如 $B / N > \text{1237}$，这是上述值的一半。你会相当频繁地看到2路数据并行，就是因为这个原因。

<p markdown=1 class="takeaway">**要点总结：** 数据并行和ZeRO分片在H100或B200上成为计算受限的条件是，每个GPU的批次大小约为2500个token（假设完美重叠和FLOPs利用率）。对于MoE模型，这个值按 $E / k$（总参数与激活参数的比率）倍增加。当进行少量数据并行时，临界批次大小会降低。</p>

### 张量并行

张量并行需要在激活值上进行AllGather和ReduceScatter操作，我们需要将这些操作与MLP的浮点运算（FLOPs）重叠。换句话说，在前向传播中，我们有：

$$T_\text{math} = \frac{2\cdot 2 \cdot BDF}{Y \cdot C}$$

$$T_\text{comms} = \frac{2\cdot 2 \cdot BD}{W_\text{collective}}$$

为了成为计算受限，我们得到规则：

$$Y < \frac{F \cdot W_\text{collective}}{C}$$

在节点内部，这给出大约 $F / 2200$ 或 $F / 2475$（跨节点）。对于像LLaMA-3中 $F=\text{28000}$ 的情况，这意味着大约11路张量并行（或向下取整，大约8路，也就是一个节点的大小）。与上面类似，当精确跨越2个节点时，我们获得额外的2倍带宽，因此我们通常可以进行16路张量并行（$F > 2475 \cdot (Y - 8)$），理论上这允许我们进行高达19路的模型并行。

<p markdown=1 class="takeaway">**要点总结：** 对于前馈维度为F、大小为Y的轴进行张量并行，当 $Y > F / 2475$ 时会变为通信受限，这通常将我们限制在仅节点内张量并行，或最多2节点张量并行。</p>

### 专家并行

正如我们上面已经提到的，混合专家（MoE）模型的模型权重增加了E倍，但浮点运算（FLOPs）只增加了k倍，这使得数据并行变得明显更加困难。我们可以通过沿着专家维度对权重进行分片来缓解这个问题，即 W<sub>in</sub>[E<sub>Z</sub>, D, F]。为了执行MLP模块，我们需要引入2次AllToAll操作，将激活值发送到对应的专家。

如上所述，如果跨越多个节点，AllToAll<sub>Z->k</sub>([B, D, k]) 的成本大致为 $T_\text{AllToAll} = 2 \cdot B \cdot D \cdot (Z-8)/Z \min(8 * k / Z, 1)$，因此对于纯专家并行，我们需要：

$$T_\text{math} = \frac{4 \cdot B \cdot k \cdot D \cdot F}{Z \cdot C}$$

$$T_\text{comms} = \frac{4 \cdot B \cdot D \cdot (Z-8)}{W \cdot Z} \cdot \min\left(\frac{8 \cdot k}{Z}, 1\right)$$

我们需要 $K > Z/8$ 且 $F > \alpha \cdot (Z - 8)/k$，或者 $Z \gg K$ 且 $F > 8 \cdot \alpha$，其中 $\alpha = C/W$。这为你提供了两个可以进行专家并行的领域：一个是少量的专家并行（大约2节点）和较小的 $F$；另一个是较大的 $F$ 和任意大的 $Z$（最高可达E路专家并行）。

在实践中你会看到这两种情况，要么是少量的专家并行（像DeepSeek v3，其F非常小，并且跨节点专家并行相对较小且受限），要么是具有较大F的模型，在这种情况下我们可以进行显著的跨节点专家并行（EP），同时配合张量并行（TP）。

<p markdown=1 class="takeaway">**要点总结：** 如果 $F < 8 * C / W_\text{node}$，专家并行可以跨越1-2个节点，其成本与张量并行相似（略低）；或者如果 $F > 8 * C / W_\text{node}$，我们可以进行大量的专家并行（最多 $E$ 个节点），且成本相对较低。</p>

### 流水线并行

流水线并行将层跨节点拆分，通信成本极低，因为我们每隔几层只发送小型微批次（microbatch）的激活值。历史上，流水线饱受"流水线气泡（pipeline bubbles）"困扰，但随着新的零气泡流水线方法的出现，通常可以避免这个问题。

流水线的整体通信成本很小：拥有 $N_\text{MB}$ 个微批次和 $N_\text{stages}$ 个阶段时，我们有 $T_\text{comms per hop} = 2 \cdot B \cdot D / (W \cdot N_\text{MB})$ 以及 $N_\text{MB} + N_\text{stages} - 2$ 次跳转，因此大致为：

$$T_\text{total PP comms} = \frac{2BD}{W \cdot N_\text{MB}} \cdot (N_\text{MB} + N_\text{stages} - 2)$$

$$T_\text{per-layer comms} \approx 1.5 \cdot \frac{2BD}{W \cdot N_\text{layers}}$$

由于我们除以 $N_\text{layers}$，这比任何其他成本都要小得多。换句话说，从通信的角度来看，流水线基本上是免费的。那么，为什么我们不直接使用流水线呢？有几个原因：

(1) **代码复杂性：** 流水线不像其他方法那样很好地融入自动并行框架（如XLA的GSPMD）。因为它引入了微批处理来隐藏流水线气泡，它改变了程序的结构，而自定义的零气泡流水线调度通过要求复杂的前向和反向传播交错进一步加剧了这个问题。

(2) **流水线使数据并行和FSDP变得困难：** 不使用流水线的可能最大原因是它与FSDP和数据并行配合不佳。特别是ZeRO-3分片效果很差，因为它要求我们在每个微批次上都进行权重AllGather，当我们只有 $B / N_\text{microbatches}$ 个token来摊销AllGather成本时，这是行不通的。此外，在反向传播期间，*我们无法在最后一个微批次通过某个阶段之前对梯度进行AllReduce或ReduceScatter，这意味着我们有显著的未重叠通信时间。*

{% include figure.liquid path="assets/gpu/pipeline-bubble.png" class="img-fluid" caption="<b>图：</b>一个2阶段、2微批次的流水线示例。F表示一个阶段的前向传播，B表示一个阶段的反向传播（成本是2倍）。G表示数据并行的AllReduce操作，其时间可能显著长于单个微批次的时间。" %}

(3) **流水线气泡和步骤失衡：** 如你在上面的（糟糕）流水线调度中所见，在一个朴素的流水线调度中很容易出现显著的气泡（意味着计算浪费）。上面的例子中，第二阶段在步骤0空闲，第一阶段在步骤2到3空闲，第二阶段在最后一个步骤再次空闲。虽然我们可以通过仔细调度在一定程度上避免这些，但我们通常仍会有一些气泡。我们还必须在关键路径上将激活值从一个阶段传递到下一个阶段，这可能会增加开销：

{% include figure.liquid path="assets/gpu/pipeline-transfer.png" class="img-fluid" caption="<b>图：</b>一个流水线示例，红色显示传输成本。这使得各阶段相对于彼此发生偏移，并增加了流水线气泡开销。" %}

对于这些问题都有解决方法，但它们往往实现复杂且难以维护；流水线仍然是一种通信成本相对于其他方法较低的技术。

**关于延迟的注意事项：** 如前所述，即使消息相当大，GPU也难以实现完整的AllReduce带宽。这意味着即使我们理论上可以跨多个节点扩展例如专家并行的AllToAll，我们也可能难以达到总带宽的50%。这意味着我们确实尝试将张量并行（TP）或专家并行（EP）保持在较少数量的节点内，以最小化延迟开销。

### 示例

**DeepSeek是怎么做的？** 作为参考，[DeepSeek V3](https://arxiv.org/abs/2412.19437) 使用2048个H800 GPU进行训练，其配置为：

* 64路专家并行（EP），跨越8个节点
* 16路流水线并行（PP）
* 2路ZeRO-1数据并行（DP）

他们的稳态批次大小为 `4096 * 15360 = 62,914,560` 个token，即每GPU 30k个token。你可以看到这已经相当大了，但他们的模型也非常稀疏（k=8, E=256），因此需要相当大的批次大小。你可以看到，使用64路EP和16路PP，我们总共得到了1024路模型并行，这意味着AllReduce是在spine级别完成的，并且由于只有2路，我们实际上获得了 $2 / (2 - 1) = 2$ 倍的带宽。这也有助于降低最终数据并行AllReduce与最终流水线阶段重叠的成本。

**LLaMA-3是怎么做的？** LLaMA-3在16k GPU上以16M token的批次大小进行训练，即大约每GPU 1k token。他们使用了：

* 节点内8路张量并行（TP）
* 16路流水线并行（PP）
* 128路ZeRO-1数据并行

由于这也是一个稠密模型，因此总的来说这些操作相当简单。16路PP将数据并行AllReduce的成本降低了16倍，这有助于我们降低临界批次大小。

### GPU上LLM扩展的总结

让我们退一步，对目前学到的内容进行一个总体概述：

* **数据并行或FSDP（ZeRO-1/3）需要每个GPU大约2500个token的本地
## Acknowledgements and Further Reading
本章内容得到了众多GPU专家的大力帮助，特此致谢：

* 亚当·帕什克（Adam Paszke），他帮助阐述了GPU内核编程（Kernel Programming）的实际运作机制。
* 斯瓦普尼尔·帕蒂尔（Swapnil Patil），他首次解释了GPU网络互联（GPU Networking）的工作原理。
* 斯塔斯·贝克曼（Stas Bekman），他指出GPU的实际性能表现往往与标称规格存在差异。
* 莱纳·波普（Reiner Pope），他帮助厘清了GPU与TPU在硬件层面的比较。
* 弗雷德里克·巴斯蒂安（Frédéric Bastien），他对芯片层级的技术细节提供了详尽的反馈。
* 努阿曼·塔齐（Nouamane Tazi），他在GPU上进行大语言模型（LLM）训练的经验完善了Roofline分析部分。
* 桑福德·米勒（Sanford Miller），他帮助我理解GPU的网络互联方式以及NVIDIA的规格说明与实际部署场景的对比。

关于GPU的优秀读物很多，以下是我个人推荐的一些：

* [SemiAnalysis的NVIDIA Tensor Core演进史](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)：一篇精彩的文章，详细描述了GPU如何从游戏引擎演变为机器学习加速器。
* [SemiAnalysis的Blackwell性能分析](https://semianalysis.com/2024/04/10/nvidia-blackwell-perf-tco-analysis/)：值得一读，有助于理解下一代NVIDIA GPU。
* [H100 DGX SuperPod参考架构](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)：虽然略显枯燥，但对于理解大规模GPU集群的网络互联方式很有用。[这里](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)有一份关于GB200系统的类似文档。
* [Hot Chips会议关于NVLink交换机的演讲](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)：有趣的资料，介绍了NVLink和NCCL集合通信，特别是网络内归约（in-network reduction）。
* [DeepSeek-V3技术报告](https://arxiv.org/pdf/2412.19437)：一份优秀的大型半开源大语言模型训练报告范例，描述了其分片配置（Sharding Setup）的选取过程。
* [如何优化CUDA矩阵乘法](https://siboehm.com/articles/22/CUDA-MMM)：一篇出色的博客，详细讲解了如何使用CUDA核心（CUDA Cores）实现高效的矩阵乘法，并重点关注GPU上的缓存一致性（Cache Coherence）。
* [HuggingFace超大规模训练指南](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：一份关于GPU上大语言模型并行（LLM Parallelism）的指南，是本章的部分灵感来源。
* [从第一性原理出发让深度学习运行如飞](https://horace.io/brrr_intro.html)：一份更侧重GPU和PyTorch的大语言模型Roofline与性能工程教程。
* [康奈尔大学GPU架构理解网站](https://cvw.cac.cornell.edu/gpu-architecture)：一本与本书类似的指南，更具体地比较了GPU与CPU的内部结构。
## Appendix A: How does this change with GB200?
Blackwell架构引入了一系列重大的网络改进，包括**NVLink 5**，其总NVLink带宽提升了一倍（900GB/s）。B200仍然采用8 GPU节点，与H100相同，但**GB200系统**（结合了B200 GPU与Grace CPU）则引入了更大的NVLink域（NVL72为72个GPU，理论上可达576个）。这种更大的NVLink域也有效提升了节点的出口带宽，从而降低了节点层级以上的**集合通信成本**。

{% include figure.liquid path="assets/gpu/b200-node.png" class="img-small" caption="<b>图：</b>GB200 NVL72单元构成示意图，包含18个交换机和72个GPU。" %}

在单个节点内部，带宽的提升（从450GB/s到900GB/s）并未带来显著差异，因为每个GPU的**每秒浮点运算次数（FLOPs/s）** 也同步翻倍。我们的**理论带宽上限（Roofline）** 基本保持不变，尽管由于NVLink带宽大幅提高，**专家并行（Expert Parallelism）** 变得更加容易。

在节点之外，变化则更为显著。下面是一张来自[此处](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/network-fabrics.html#compute-fabric-576)的**SuperPod**示意图。

{% include figure.liquid path="assets/gpu/gb200-superpod.png" class="img-fluid" caption="<b>图：</b>由576个GPU组成的GB200 DGX SuperPod示意图。" %}

如图所示，单个节点的出口带宽提升至 `4 * 18 * 400 / 8 = 3.6TB/s`，而H100系统中仅为400GB/s。由于芯片的**每秒浮点运算次数（FLOPs）** 也同步翻倍，这使得有效的跨节点**理论带宽上限**提升了约4倍。现在我们可能开始需要担心，性能瓶颈是出现在节点层级，还是出现在**扩展（Scale-out）** 层级。

**Grace Hopper:** NVIDIA还销售**GH200**和**GB200**系统，它们将一定数量的GPU与一个Grace CPU配对。例如，GH200包含1个H200 GPU和1个Grace CPU，而GB200系统则包含2个B200 GPU和1个Grace CPU。这种系统的一个优势在于，CPU通过全带宽的NVLink连接（称为**NVLink C2C**）与GPU相连，因此您拥有极高的CPU到GPU带宽，这对于将参数**卸载到主机内存**非常有用。换言之，对于任何给定GPU，访问主机内存的带宽与访问另一个GPU的**高带宽内存（HBM）** 带宽是相同的。
## Appendix B: More networking details
这是NVLink 4开关的示意图。它共有64个NVLink4端口（每个端口使用2个物理通道），并配备一个大型交叉开关处理通道间切换。相比之下，TPU使用带有可动态重配置镜面的光开关。

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>图：</b>单个NVLink4开关的底层视图。" %}

在每个层级上，我们都可能受限于可用链路带宽或总交换带宽。

* **节点层级：**在节点层级，我们有4 × 1.6TB/s = 6.4TB/s的NVSwitch带宽，但我们的8个GPU中每个GPU只能以450GB/s的速率向交换机发送数据，这意味着节点内部的实际峰值带宽为450e9 × 8 = 3.6TB/s（全双工）。
* **SU/叶层级：**在SU层级，我们有8个交换机以1×400 Gbps InfiniBand的方式全连接32个节点。这提供了8 × 32 × 400 / 8 = 12.8TB/s的节点出口带宽，同时我们在交换机层面有8 × 1.6TB/s = 12.8TB/s的带宽，两者完全吻合。
* **脊层级：**在脊层级，我们有16个交换机通过2×400 Gbps链路连接32个叶交换机，因此我们有32 × 16 × 400 × 2 / 8 = 51.2TB/s的出口带宽。这16个交换机提供16 × 1.6TB/s = 25.6TB/s的带宽，所以这是该层级的瓶颈。

按每个GPU计算，节点层级的GPU到GPU带宽为450GB/s，SU层级为50GB/s，脊层级为25 GB/s。

**GPU经验性AllReduce带宽：**

{% include figure.liquid path="assets/gpu/gpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>8xH100集群的AllReduce带宽（节点内，禁用SHARP）。" %}

TPU v5p带宽（1个轴）：

{% include figure.liquid path="assets/gpu/tpu-all-reduce-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5p 4x4x4集群的AllReduce带宽（沿一个轴）。" %}

以下是AllGather带宽：

{% include figure.liquid path="assets/gpu/gpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>8xH100集群的AllGather带宽（节点内）。" %}

{% include figure.liquid path="assets/gpu/tpu-all-gather-bw.png" class="img-fluid" caption="<b>图：</b>TPU v5e 8x16集群的AllGather带宽（沿一个轴）。" %}

**关于AllToAll成本的更多信息：**

在此，我们可以比较近似值 $\min(K / Z) * (Z - 1) / Z$ 与真实值 $(1 - ((Z - 1) / Z) ** K) * (Z - 1) / Z$。除了在 $Z$ 值较小的情况下，两者非常接近。

{% include figure.liquid path="assets/gpu/all-to-all-approx.png" class="img-fluid" caption="<b>图：</b>随着分片数量的增加，不规则AllToAll的近似成本与真实成本的比较。" %}