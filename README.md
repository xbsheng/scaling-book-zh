# 如何扩展你的模型

**基于 JAX/TPU 的大模型扩展系统指南**

本项目是 [jax-ml/scaling-book](https://github.com/jax-ml/scaling-book) 的中文翻译版本。

## 关于本书

本书旨在揭开在 TPU 上扩展 LLM 的神秘面纱。我们试图解释 TPU 的工作原理、LLM 如何在实际规模上运行，以及如何在训练和推理期间选择并行化方案以避免通信瓶颈。

在线阅读：https://xbsheng.github.io/scaling-book-zh

## 目录

| 章节 | 内容 |
|------|------|
| [引言](index.md) | 为什么需要了解模型扩展 |
| [Roofline 分析](roofline.md) | 硬件性能上限分析 |
| [Transformer 架构](transformers.md) | Transformer 的计算与通信特性 |
| [训练](training.md) | 大模型训练的并行策略 |
| [推理](inference.md) | 大模型推理优化 |
| [TPU 架构](tpus.md) | TPU 硬件详解 |
| [GPU 架构](gpus.md) | NVIDIA GPU 硬件详解 |
| [分片策略](sharding.md) | 张量分片与数据并行 |
| [性能分析](profiling.md) | 模型性能分析工具 |
| [JAX 实践](jax-stuff.md) | JAX 框架使用技巧 |
| [训练应用](applied-training.md) | 实际训练案例 |
| [推理应用](applied-inference.md) | 实际推理案例 |
| [总结](conclusion.md) | 总结与延伸阅读 |

## 本地运行

```bash
# 安装依赖
brew install imagemagick ruby  # macOS
pip install jupyter

# 克隆并运行
git clone https://github.com/xbsheng/scaling-book-zh.git
cd scaling-book-zh
bundle install
bundle exec jekyll serve
```

访问 http://127.0.0.1:4000/scaling-book-zh 查看本地预览。

## 致谢

本书由 Google DeepMind 的 Jacob Austin、Sholto Douglas、Roy Frostig、Anselm Levskaya 等人撰写。

中文翻译由 AI 辅助完成，如有问题欢迎提交 Issue 或 PR。

## 引用

```bibtex
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and others},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```

## License

本项目遵循原项目的 [Apache 2.0 License](LICENSE)。
