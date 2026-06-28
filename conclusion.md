---
layout: distill
title: "Conclusions and Further Reading"
# permalink: /main/
description: "Thank you for reading! Here we'll include a few more references for further study."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff"
previous_section_name: "Part 10: JAX"

next_section_url: "../gpus"
next_section_name: "Part 12: GPUs"

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
  - name: "Acknowledgments"
  - name: "Further Reading"
  - name: "Feedback"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---**感谢您阅读全文，并祝贺您坚持到了最后。** 在结束之前，需要特别感谢以下各位：
## Acknowledgments
本文档凝聚了Google DeepMind众多成员的集体智慧，在此我们向各位致以诚挚感谢！

- James Bradbury、Reiner Pope与Blake Hechtman最先提出了本文稿中的诸多核心观点，并较早洞察了Transformer（转换器）的系统架构理念。
- Sholto Douglas撰写了本文档初版并发起该项目，对文档整体叙事框架的构建尤为关键。
- Jacob Austin主导了将初稿笔记转化为更完善系统化成果的工作，承担了大量编辑、格式调整与文档发布工作，并协调了其他作者的贡献。
- 多数图表与动画由Anselm Levskaya与Charlie Chen制作完成。
- Charlie Chen撰写了推理章节并绘制了多幅推理相关图表。
- Roy Frostig在发布、编辑及其他诸多环节提供了重要支持。

我们同样感谢在整个过程中给予关键反馈的各位同仁，特别是Zak Stone、Nikhil Sethi、Caitlin Stanton、Alek Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins等众多贡献者。感谢Ruiqi Gao协助完成HTML格式调整。

**衷心感谢各位！**

<p markdown=1 class="announce">在离开之前，您或许也有兴趣阅读关于NVIDIA GPU（图形处理器）的新章节[第12章](../gpus)！</p>
## Further Reading
以下是一些相关的文献资料：

- [**TPU 深入解析**](https://henryhmko.github.io/posts/tpu/tpu.html)：以本书精神对TPU架构进行的精彩深入剖析。
- [**AI推理领域专用架构**](https://fleetwood.dev/posts/domain-specific-architectures)：本书精神下的硬件与模型深度解读。
- [**用于训练深度神经网络的领域专用超级计算机**](https://dl.acm.org/doi/pdf/10.1145/3360307)：最早的TPU论文之一，包含许多关于Google TPU项目的精彩细节，部分内容本书未涉及。
- [**从第一性原理看深度学习加速**](https://horace.io/brrr_intro.html)：更侧重GPU和PyTorch的LLM算力上限与性能优化教程。
- [**使用Pallas编写TPU内核**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：TPU编程正越来越多地涉及用Pallas编写自定义内核。本系列探讨如何编写内核，以及许多本书未提及的底层TPU细节。
- [**如何优化CUDA矩阵乘内核以达到cuBLAS性能：实践记录**](https://siboehm.com/articles/22/CUDA-MMM)：虽然专注于GPU和CUDA，但这篇优秀的博文展示了如何优化CUDA矩阵乘内核，有助于深入理解TPU与GPU的差异。
- [**分布式数组与自动并行化**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：关于JAX并行化API的精彩指南，有助于实践本书讨论的一些理念。
- [**Rafi Witten的2024年高性能LLM课程**](https://github.com/rwitten/HighPerfLLMs2024)：我们的前同事Rafi开设了精彩的TPU性能优化课程，幻灯片均在GitHub上。其中许多内容比本书探讨得更深入。
- [**\[2211.05102\] 高效扩展Transformer推理**](https://arxiv.org/abs/2211.05102)：详细阐述Transformer推理数学原理的论文，是本文档的重要灵感来源。
- [**Huggingface超大规模实践手册**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：本书在GPU领域的姊妹篇，更深入探讨PyTorch在训练中如何实现并行化技术与内存优化技术。
- [**Transformer推理算术**](https://kipp.ly/transformer-inference-arithmetic/)：包含许多与本书相同理念的博客，并配有精美图解。
- [**斯坦福CS336幻灯片与视频**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：斯坦福大学关于LLM训练与部署细节的精彩课程，包含实用练习，其中作业1和2尤为相关。
- [**Stas Bekman的机器学习工程手册**](https://github.com/stas00/ml-engineering)：高度实用的机器学习基础设施指南，涵盖本书未涉及的主题，如如何与云服务商谈判、集群管理以及GPU吞吐量的实证测量。

该领域仍有广阔的综合论述空间，我们期望本文稿能激发更多此类著述！我们亦相信这是一个值得研究与探索的沃土，即使手头没有众多硬件加速器，也能开展许多研究工作。
## Feedback
欢迎留下评论或问题，以便我们进一步改进。您可以通过jacobaustin123 [at] gmail [dot] com联系通讯作者Jacob Austin，或在GitHub上通过提交问题、拉取请求（pull requests）或发起讨论（discussions）[来建议修改](https://github.com/jax-ml/scaling-book)。