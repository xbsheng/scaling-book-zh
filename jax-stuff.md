---
layout: distill
title: "Programming TPUs in JAX"
# permalink: /main/
description: "How to use JAX to program TPUs efficiently! Much of this section is taken from <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>here</a>. You can run the code examples in this section with free TPUs on <a href='https://colab.sandbox.google.com/'>Google Colab</a>."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling"
previous_section_name: "Part 9: Profiling"

next_section_url: ../conclusion
next_section_name: "Part 11: Conclusions"

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
  - name: Yash Katariya
    url: https://x.com/yashk2810
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "How Does Parallelism Work in JAX?"
  - subsections:
    - name: "Auto sharding mode"
    - name: "Explicit sharding mode"
    - name: "Manual sharding mode via shard_map"
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

## How Does Parallelism Work in JAX?
JAX 支持三种多设备编程理念：

1.  **编译器，你来掌控！** 让 XLA 编译器自动对数组进行分区并决定添加何种通信以满足特定程序的需求。这使您无需修改任何代码，即可将在单个设备上运行的程序自动运行到数千个设备上。
2.  **JAX，你来掌控！** 自动并行化很棒，但有时编译器会做出奇怪的事情。显式分片让您能像往常一样编写单设备代码，但由 JAX 来处理分片传播（而非编译器）。这意味着当 JAX 不清楚您的意图时，它可以向您询问以明确。
3.  **让我直接写我想写的意思！** 虽然编译器很方便，但它们有时会做错事，添加非预期的通信。有时我们想要明确指定我们打算执行的具体通信操作。

| 模式 | 视图？ | 显式分片？ | 显式集合操作？ |
|:---:|:---:|:---:|:---:|
| 自动 | 全局 | ❌ | ❌ |
| 显式 | 全局 | ✅ | ❌ |
| 手动 | 逐设备 | ✅ | ✅ |

相应地，JAX 为每种模式提供了 API：

1.  `jax.jit`（配合 `Auto` 网格轴）允许您获取任何现有的 JAX 函数，并用分片输入调用它。然后 JAX 使用 XLA 的 [Shardy](https://openxla.org/shardy) 编译器来自动并行化程序。XLA 将在需要时自动为您添加通信（如 AllGather、ReduceScatter、AllReduce 等）以支持现有操作。虽然它并不完美，但通常能很好地将您的程序自动扩展到任意数量的芯片，而无需修改代码。
2.  使用 `Explicit` 网格轴的 `jax.jit` 看起来与（1）类似，但让 JAX（而非 XLA）来处理分片传播。这意味着数组的分片实际上是 JAX 类型系统的一部分，当 JAX 检测到模糊的通信时，它可以报错并让用户来解决。
3.  `jax.shard_map` 是更手动的对应方式。您获得程序的设备本地视图，并且必须显式编写您想要的任何通信。有一个分片数组并希望每个设备都拥有完整内容？添加一个 `jax.lax.all_gather`。希望对数组进行跨设备求和？添加一个 `jax.lax.psum`（即 AllReduce）。编程难度更高，但做出非预期操作的可能性也小得多。

<h3 id="auto-sharding-mode">自动分片模式</h3>

`jax.jit` 在 JAX 内部扮演两个角色。顾名思义，它将函数“即时”编译为字节码（通过 XLA/HLO/LLO），从而加快运行速度。但如果输入是分片的，或者用户指定了 `in_sharding` 或 `out_sharding`，它还会让 XLA 将计算分布在多个设备上，并根据需要添加通信。例如，下面展示了如何使用 `jax.jit` 编写一个分片矩阵乘法：

```py
import jax
import jax.numpy as jnp

# 运行在 TPU v5e 4x2 上。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 这告诉 JAX 对所有操作使用此网格，因此您只需指定 PartitionSpec P。
jax.set_mesh(mesh)

# 我们创建一个矩阵 W 和输入激活值 In，它们在设备间分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=jax.NamedSharding(mesh, jax.P('Y', None)))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 我们可以在此显式编译分片的矩阵乘法函数。这会添加所有必要的通信（例如矩阵乘法后的 AllReduce）。
jit_matmul = jax.jit(matmul_square, out_shardings=jax.P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

这将根据任何分片方式自动运行，并将计算划分到我们的设备上。**但在硬件层面实际发生了什么？**

1.  首先，我们创建在设备间分片的 In 和 W <d-footnote>注意我们是如何做的。这是创建具有特定分片的数组的一种方式（即通过向创建函数添加 `device` 参数）。另一种方式是用 `jnp.array(....)` 正常创建数组，然后执行类似 `jax.device_put(..., jax.P('X', 'Y'))` 的操作。还有一种方式是编写一个创建所需数组的函数，并用 `out_shardings` 指定所需分片方式对其进行 JIT 编译。</d-footnote>。W 沿着收缩维度以 2 路方式分片，而 In 以 8 路方式分片：沿着输入维度 4 路，沿着收缩维度 2 路。这对应于分片 W[D<sub>Y</sub>, F] 和 In[B<sub>X</sub>, D<sub>Y</sub>]，即一种模型并行和数据并行的方式。
2.  如果我们在本地运行（即在单个设备上），`matmul_square` 只会将输入平方并执行简单的矩阵乘法。但因为我们指定了 `out_shardings` 为 `P('X', None)`，输出将沿着批次维度分片，但在模型维度上复制，并且需要一次 AllReduce 来计算。

使用我们前面的符号，这可能会执行如下操作：

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit` 会自动为我们添加这个！我们实际上可以通过 `jit_matmul.as_text()` 打印 HLO，并看到如下 HLO（经过大幅缩写）：

```py
# 这个融合操作是分片输入和矩阵的实际矩阵乘法
%fusion = bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[2,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# 我们跨设备对部分求和结果进行规约
ROOT %AllReduce = bf16[2,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[2,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

我们可以看到上面的矩阵乘法（融合操作）和 AllReduce。特别注意形状。`bf16[2, 1024]` 是激活值的本地视图，因为我们的 `batch_size=8` 分散在 4 个设备上，而我们的 `d_model=2048` 同样以 2 路方式分片。

**这非常神奇！** 无论我们的程序多么复杂，[Shardy](https://openxla.org/shardy) 和 JIT 都会尝试为所有中间激活值找到分片方式，并根据需要添加通信。话虽如此，Shardy 也有其缺点。它可能会犯错。有时您会查看性能分析图，注意到某些地方出了问题。一个巨大的 AllGather 占据了性能分析的 80%，而它本不需要这样。当这种情况发生时，我们可以尝试通过使用 `jax.lax.with_sharding_constraint` 显式注释中间张量来纠正编译器。例如，对于两个矩阵乘法，我可以通过以下方式强制中间激活值沿着 `y` 维度分片（虽然这可能不是个好主意）：

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))
jax.set_mesh(mesh)

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.P('X', 'Y'))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

这大约构成了自动分区世界中 JAX 并行编程的 60%，您通过 `jax.lax.with_sharding_constraint` 控制中间分片。但“调戏编译器”众所周知不是一个愉快的编程模型。您可以注释每个中间变量，但仍然不知道是否会得到正确结果。那么，如果 JAX 本身能够处理和控制分片传播呢？

<h3 id="explicit-sharding-mode">显式分片模式</h3>

显式分片（或“类型中的分片”）看起来很像自动分片，但分片传播发生在 JAX 层！每个 JAX 操作都有一个分片规则，该规则接收操作参数的分片并生成操作结果的分片。您可以使用 `jax.typeof` 查看结果分片：

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

# 运行在 TPU v5e 2x2 上。这为硬件的两个物理轴分配名称。
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'),
                                       axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))

# 这告诉 JAX 对所有操作使用此网格，因此您只需指定 PartitionSpec P。
jax.set_mesh(mesh)

x = jax.device_put(np.arange(16, dtype=np.float32).reshape(8, 2), jax.P('X', 'Y'))

@jax.jit
def f(x):
  print(jax.typeof(x))  # float32[8@X,2@Y]
  out = x * 2
  print(jax.typeof(out))  # float32[8@X,2@Y]
  return out

f(x)
```

如您所见，JAX 将分片从输入 (`x`) 传播到输出 (`out`)，并且可以在追踪时通过 `jax.typeof` 检查。对于大多数操作，这些规则简单明了，因为只有唯一合理的选择（例如，元素级操作保持相同的分片）。但对于某些操作，如何对结果进行分片是模糊的，在这种情况下，JAX 会抛出追踪时错误，并要求程序员显式提供 `out_sharding` 参数（例如 jnp.einsum、jnp.reshape 等）。让我们再看一个存在冲突的例子：

```py
# 我们创建一个矩阵 W 和输入激活值 In，它们在设备间分片。
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=jax.P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=jax.P('Y', None))

@jax.jit
def matmul_square(In, W):
  print(jax.typeof(In))  # bfloat16[8@X, 2048@Y]
  print(jax.typeof(W))  # bfloat16[2048@Y, 8192]
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 这会报错
```

此代码会报错，信息为：

```
Contracting dimensions are sharded and it is ambiguous how the output should be sharded.
Please specify the output sharding via the `out_sharding` parameter.
Got lhs_contracting_spec=('Y',) and rhs_contracting_spec=('Y',)
```

这很棒，因为 einsum 输出应如何分片是模糊的。输出分片可以是：
* P('X', 'Y')，这将引发 ReduceScatter，或者
* P('X', None)，这将引发 AllReduce

与自动模式不同，显式模式在检测到模糊通信时会报错，并要求用户解决它。所以这里您可以这样做：

```py
@jax.jit
def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W, out_sharding=jax.P('X', 'Y'))

out = matmul_square(In, W)
print(jax.typeof(out))  # bfloat16[8@X,8192@Y]
```

自动模式和显式模式可以通过 `jax.sharding.auto_axes` 和 `jax.sharding.explicit_axes` API 组合使用。这是一个[很棒的文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)可以阅读获取更多信息。

<h3 id="manual-sharding-mode-via-shard_map">通过 shard_map 的手动分片模式</h3>

虽然 Shardy 是“编译器掌控”模式，但 JAX [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html) 则将一切交到您手中。您像在 jax.jit 中一样指定输入的分片，然后显式编写所有通信。`jax.jit` 为您提供程序的全局跨设备视图，而 `shard_map` 则为您提供本地的逐设备视图。

这里有一个例子。试着思考这个函数的作用：<d-footnote>如果您想在 Colab 中通过模拟网格自己尝试，可以使用以下单元格：`import jax; jax.config.update('jax_num_cpu_devices', 8)`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

mesh = jax.make_mesh((2, 4), ('x', 'y'), (shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=jax.P(('x', 'y')))

# 此函数将在数组的 1/8 部分上操作。
@jax.shard_map(in_specs=jax.P(('x', 'y')), out_specs=jax.P())
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = slice_and_average(x)
assert out.shape == (4,)
```

**这是做什么的？** `slice_and_average` 在每个 TPU 上运行，处理数组的 1/8，我们从中切片前 4 个元素，并在整个网格上进行平均。这意味着我们实际上是在计算 `mean(x[:4], x[64:68], x[128:132], …)`。这很酷，因为否则在 JAX 中表达这样的操作并不容易。

**为什么要用这个而不是 jax.jit？** 如果我们使用 `jax.jit`，`slice_and_average` 会看到数组的全局视图（完整的 `[512,]` 数组）。我们得切出这个不规则的片段，然后执行一个平均操作，而 XLA 需要正确地解释它。XLA 可能会添加错误的通信或感到困惑。在这里，我们看到的是本地视图，并且只编写我们需要的通信。

**示例 [集合矩阵乘法]：** 举一个更现实的例子，假设我们要实现模型并行，其中激活值最初是模型分片的，即 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]。简单来说，我们会通过先对 A 进行 AllGather，然后进行本地矩阵乘法来实现：

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>])
2. Out[B<sub>X</sub>, F<sub>Y</sub>] =
## Worked Problems
以下是关于JAX的一些随机问题，后续我还会补充更多。所有这些任务都需要在Colab中使用一定数量的TPU。你可以使用配备TPU v2-8的公共Colab。从现在开始，我们将假设你拥有N个可用设备。

**问题1：** 设**A**是一个形状为float32[S<sub>X</sub>, D<sub>Y</sub>]的激活值数组，其中`X * Y = N`。请完成以下任务：

1. 用JAX编写一个函数，计算每个`(X, Y)`分片内的平均值，即返回一个大小为[X, Y]的数组，其中`arr[i, j]`是分片`(i, j)`的平均值。分别使用`jax.jit`和`shard_map`实现，并对每种实现进行性能分析，看看它们各耗时多久。是否增加了通信开销？*提示：理论上不应该有，但有时XLA还是会添加通信。*

2. 用JAX编写一个函数，对**X轴方向每个分片内**的某个偏移量**返回`roll(x, shift, axis=0) - x`**。我还没那么自虐到让你用jax.jit实现这个，所以用`shard_map`完成即可。

<details><summary>点击查看答案。</summary>


第1部分：这里是第1部分的解决方案。注意，为了`jax.jit`的解决方案，我们必须进行相当复杂的重塑操作。

```py
import numpy as np

import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

第2部分：这里是第2部分的类似解决方案。

```py
import numpy as np

import jax
import jax.numpy as jnp

import functools

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = jax.shard_map(
      lambda x: jnp.roll(x, shift, axis=0),
      mesh=mesh,
      in_specs=jax.P('X','Y'), out_specs=jax.P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, jax.P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.float32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, jax.P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```


</details>

**问题2：** 现在我们来一起构建一个基本的“专家混合”模型。设**W**: float32[E<sub>X</sub>, D, F]是一组E个“专家”矩阵。设**A**: float32[S<sub>X</sub>, D]（我们的激活值），设**B**: int32[S<sub>X</sub>]是一组“路由分配”，其中B[i]是范围`[0, E)`内的一个整数，告诉我们想要用哪个矩阵来处理该激活值。我们希望用JAX编写一个函数，返回`Out[i] = A[i] @ W[B[i]]`。

1.  首先我们完全忽略分片。将所有这些张量设置得足够小，以便能放入单个设备。编写这个函数的本地实现。*确保你不要具体化一个形状为`[S, D, F]`的数组！提示：尝试将令牌（tokens）排序到一个形状为`[E, S, D]`的新缓冲区中，并注意掩码（masking）处理（为什么我们需要第二个维度的大小为S？）。*

2.  如果你直接对上述方法使用`jax.jit`，会发生一些事情。对其进行性能分析，看看它决定执行了什么通信。耗时多久？

3.  你会注意到上述实现的一个问题是，它可能在本地收集了完整的激活值集合**A**，即AllGather<sub>X</sub>([S<sub>X</sub>, D])。这不仅在通信开销上很昂贵，如果我们无法在本地容纳完整的激活值集合，在内存开销上也是极其昂贵的。使用`shard_map`和显式通信来实现上述功能。

      1.  第一种尝试，最简单的方法可能是使用`jax.lax.all_gather`并像步骤1那样重新排序。

      2.  第二种尝试，尝试避免具体化任何大小为`[E, S, D]`的数组，即尝试在`jax.lax.while_loop`内部使用`jax.lax.all_to_all`以不规则（ragged）方式执行计算。这样，你可以避免具体化完整的激活值，并避免在填充（padding）上浪费计算。这比你最初的实现快多少？

4.  大多数专家混合模型（MoEs）会路由到多个（k个）专家，然后对结果进行平均。重构上述实现来完成这个功能。在这种情况下，设**B**: int32[S<sub>X</sub>, k]表示要路由到的k个专家。

<details><summary>点击查看（部分）答案。</summary>


1/2. 对于第(1)部分，你有很多选择。这里有一个选项，只是用掩码遍历专家。

```py
def moe_local(W: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry  # [S, F]
        mask = (B == e)[:, None]  # [S, 1]
        expert_result = A @ W[e]  # [S, F] - 该专家对所有令牌的变换
        output = output + expert_result * mask  # 仅保留分配给该专家的结果
        return output, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))

    return output
```

你也可以使用`jax.lax.ragged_dot`，它会执行类似的操作，但更高效。

3. 我这里只概述伪代码（如果你有简洁的解决方案，请随意添加）：

```py
chunk_size = 128
def matmul(W, x, B):
  i = 0
  x = # 根据分配对x进行排序
  while (chunk := x[i:i+chunk_size]).any():
     chunk = all_to_all(chunk)
     out = matmul_local(W, chunk)
     i += chunk_size
  return concat(out)
```

基本思路是遍历数组的分块，对它们进行排序并执行all_to_all操作，然后进行本地浮点运算。


</details>

**问题3：** 上面的集体矩阵乘法示例实际上对真实的LLM非常相关。让我们调整这个示例来实现完整的Transformer堆栈。

1.  作为练习，我们首先实现一个AllReduce集体矩阵乘法，即A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]。注意输出没有被复制。朴素的算法在上面已经讨论过，基本上就是一个本地矩阵乘法后跟一个AllReduce。尝试制作一个通信重叠的“集体”版本的操作。*提示：在输出维度上进行分块（tile），并随意使用`jax.lax.psum`（即AllReduce）。* *注意：由于XLA的处理方式，它可能实际上并不比基线快。*

2.  上面AllReduce集体矩阵乘法的互补操作是ReduceScatter集体矩阵乘法，如Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]。这出现在Transformer的降维投影矩阵中。在JAX中实现一个集体的、重叠的版本。注意只传递你所需的最少数据量。*提示：尝试在累积结果时对其进行置换。*

3.  将这两个操作组合成一个端到端的Transformer块，执行In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]，并实现通信重叠。<d-footnote>和之前一样，我们不能先计算$W_{in} \cdot W_{out}$，因为这里省略了一个非线性操作。</d-footnote> 这比`jax.jit`实现快多少？

**问题4：** 上面实现的所有集体矩阵乘法都是单向的：它们只在一个方向上进行置换。重写集体AllReduce矩阵乘法和集体ReduceScatter矩阵乘法，使其使用双向通信。这些快了多少？

### 这就是第10部分的全部内容。基本上就是这样了！如需最终结论和进一步阅读，请点击[这里](../conclusion)。