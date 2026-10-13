# TerraCUDA 项目接手指南

> 写给下一个会话的我。mod 的契约只有一句话：**NOISE 阶段在 GPU 上、与 vanilla 逐位一致，
> 其余十一个阶段一个字节不碰。** 这份指南是唯一需要读的东西——读完就能干活。

## 一句话定位

NeoForge mod（MIT），把 Minecraft 26.1.2 的 NOISE 阶段（地形密度 → 方块）卸载到 NVIDIA
GPU（CUDA），与 vanilla 逐位一致；其余阶段全部原样走 vanilla（包括它们的调度）。
MC 26.1.2 / NeoForge 26.1.2.114 / Java 25 / `mod_id=terracuda`。远端 `github.com/tqk114514/TerraCUDA`。

## 当前数字

- **同日三场背靠背（2026-10-13，Chunky 22,801，种子 123，机器空闲，cap 50 声明）**：
  vanilla **4:54**（77.6 cps）→ NOISE-only **4:26**（85.7 cps，**1.11×**）→
  **+offload（L1+L3）2:17**（166.4 cps，**2.15×**）。offload 场 GPU busy 72%、噪声窗口
  186.9/s——管线终于喂饱设备线程。性能笔记 §15 有全表与判读。
- **范围结构**：`-Poffload` 是**默认关的可选性能包**（半径 0 阶段体下放 + FEATURES 专用
  线程，性能笔记 §15 / 26.1.2 管线调研报告）；不带它的 mod 形态 = 纯 NOISE 接管。
- **对照**：zFastNoise（26.3，同契约、同机、其自家 JMH 工具）阶段体 **1.46×**（隔离、
  单线程、float 栈上的 11.53→7.88 ms/区块）——口径不同不可直接比，见性能笔记第十四节。
- **26.3 的现实**：float 栈把 vanilla 阶段体压到单线程 ~11.5 ms，GPU 卸载的价值空间
  比 26.1 小一个量级——26.3 适配的立项依据要先重测。调度层工作（L1/L3 及后续 L2）
  跨版本保值（管线类在 26.3 全部存活）。
- 性能笔记十~十二节的历史数字测于带额外实验的构建，引用前先读 §13/§15 的范围说明。

## 必读文件

| 文件 | 内容 |
|---|---|
| [`.workbuddy-ai/memory/MEMORY.md`](.workbuddy-ai/memory/MEMORY.md) | **先读这个**——精简版长期记忆 |
| [`docs/性能笔记.md`](docs/性能笔记.md) | 所有测量与决定（第一~十三节），数字以此为准 |
| [`docs/GPU_区块生成规划.md`](docs/GPU_区块生成规划.md) | 原始规划文档 |
| [`.workbuddy-ai/memory/2026-10-04.md`](.workbuddy-ai/memory/2026-10-04.md) | SURFACE 重开知识 + vanilla 内部速查 + 坑清单 |

## 已完成的优化（不需要重做）

- M0–M4 全闭环（规划文档定义的里程碑）
- P0：计时仪器（噪声计数、GPU 线程窗口、节流器状态、延迟分位数）
- P1：拆线（设备线程产密度 → replay workers 消费）
- 多流+页锁定：三段设备 pass 重叠（页锁定是前提）
- 双 rules worker，每 worker 一套解释器
- beardifier 宿主侧逐方块加 beard
- invokeExact：热路径 FFM 调用 + Enable-Native-Access manifest
- `applyRules`/`writeChunk` 方法拆分 + replay 队列 16（方法留下，拆线程实测更慢不留）
- **L1+L3 调度层扩展（`-Poffload`，默认关）**：半径 0 阶段体下放 + FEATURES 专用线程
  ——平移自实验线并收窄判据（半径 0），激活时各打一行日志作为证据
- **L2 调度 lane（`-PparallelSched`，默认关）**：阶段推进步骤上 N 条 lane——平移并验收，
  **无头协议下负结果**（端到端持平、噪声 p50 变差，§15 L2 节），保留作客户端场景的未测假设
- **逐层归因仪表（`-Ptiming` 门控，对任何形态生效）**：`LayerTiming` 三把尺——每层
  wall（首步→排空）、requeue（停泊续体往返）、task dwell；§16 有 vanilla/offload 两态解剖表

## 下一步（按性价比序）

1. **G2：规则与 aquifer 上 GPU**（唯一剩下的结构杠杆）。**G2a 已落地**（§21，
   `b3a8b7f`）：`terracuda_material_decide` 内核 + `MaterialDeciderGpu` 管道，
   6 区块 589,824 块对 CPU 全路径逐位一致（含 AIR 语义陷阱的修正），94 测试全绿。
   剩余：**G2b** 格点半场的噪声上设备（`exportGrid` 的流体状态计算——floodedness/
   spread/lava + erosion/depth flat 表 + K0 地表水位——消灭 CPU 供表）→ **G2c** 输出翻
   方块 id、relay 退化纯写回（砍停留总量的那刀）。每步 94 测试 + 逐位验收 + 同日基准。
2. **客户端场景**（未测）：节流器 4÷dwell 的世界；`-Ptiming` 逐层仪表直接可用，
   需要手动玩家操作。
3. 已封案：FEATURES 杠杆（§17）、L4 无头（§16）、L2 lane（§15）、G1 批处理-屏障
   （§19：波 14.2/波成立但屏障使阶段延迟翻倍、端到端持平；`-Pbatch` 默认关保留，
   "流水线化波"是记录在案的推断级假设）。`-PreplayCapacity=8` 延迟档（§17）。

## 关键命令

```powershell
# 构建 + 测试（91 个，全绿是底线）
./gradlew test

# 无头基准（优先用这个，不需要玩家）
# 前提：run/server.properties 开 RCON(enable-rcon=true, rcon.password=terracuda)，level-seed=123
#       run/mods 装 Chunky，run/eula.txt eula=true
Remove-Item -Recurse -Force run/world, run/chunky -ErrorAction SilentlyContinue
./gradlew runServer -Ptakeover -Ptiming -Poffload   # 后台启动；offload 是可选性能包，验收要跑两种形态
.\benchmark\chunky-rcon.ps1 -Command "chunky shape square"
.\benchmark\chunky-rcon.ps1 -Command "chunky center 0 0"
.\benchmark\chunky-rcon.ps1 -Command "chunky radius 1200"
.\benchmark\chunky-rcon.ps1 -Command "chunky start"
# 等完成（约 3-5 分钟），从 run/logs/latest.log 提取 "Total time" 和 "noise stage" 行
.\benchmark\chunky-rcon.ps1 -Command "stop"

# 带玩家验证（runClient 需要手动操作）
./gradlew runClient -Ptakeover -Ptiming
# 种子 123，视距 32，传送 (1000, 1000) 站定

# 其他开关
./gradlew runServer -PrulesWorkers=3  # 改 worker 数（诊断用）
./gradlew runServer -Ptakeover -Ptiming -PnoFeatures  # offload 开着但 FEATURES 留在 dispatcher（诊断用）
./gradlew runServer -Ptakeover -Ptiming -PreplayCapacity=8   # 延迟调优档（noise p50 −28%，§17；也可叠 -Poffload）
./gradlew runServer -Ptakeover -Ptiming -PchunkyWorking=100  # 覆盖 Chunky 在飞上限（跨研究比较必须声明 cap）
# 逐线程 busy% 采样（哪条线程是墙）：.\benchmark\thread-sampler.ps1 -Samples 4 -IntervalSeconds 10
# 跑分前查后台负载（victoria3/ffmpeg 教训）——外部污染是废数字的根源
```

## 铁律（违反 = 地形不一致）

1. CUDA 必须 `--fmad=false`，绝不能 `--use_fast_math`
2. Java 表达式逐字移植 CUDA，不"整理"结合顺序
3. `XoroshiroRandom.nextDouble()` 用 float 运算
4. `2^n` 用 `ldexp` 不用 `pow`
5. `Mth.lerp(float,float,float)` 是 float 运算；spline 全程 float
6. 自有类与 vanilla 同名，测试不 import vanilla 同名类
7. 每个原语对 vanilla 有逐位 parity 测试
8. CPU 参考实现也必须对 vanilla 验证

## 许可红线

C2ME-fabric clone 在工作区（已 gitignore）。**整仓 MIT，唯一例外
`c2me-opts-accel-opencl/` 是 ARR——不读实现、不抄结构、不抄命名。**
MIT 部分可读可参考，但一行不抄。我们对标的是 vanilla 的输出，不是别的 mod 的成绩。

## 测量教训（违反会浪费时间）

- 同配置单场端到端带宽 **±18%**——跨场比较看分段（机制性），端到端必须复跑
- busy% 含队列等待 ≠ 设备占用
- 仪表必须能看见它声称测量的改变；新仪器先对已知量校准
- 外部负载（victoria3、ffmpeg）污染一切端到端数字——跑分前先查
- 过了 thread safety 测试 ≠ 没有 bug

## 关键教训（项目方法论）

- 诊断对两个假设给出**相同**数字时，先怀疑诊断
- 消融（顶成常量再计时）比采样推断有说服力
- 每个优化前先确认它瞄准的是绑定约束
- 负优化也是数据；同样，**方向性放弃也是数据**（第十三节的收束就是一次）

## 工作流约定

- 提交信息承载"为什么"和"之前哪个结论错了"（英文正文，格式参照 git log）
- 性能数字进 `docs/性能笔记.md`，提交信息不重复数字
- pwsh 不支持 heredoc——提交信息用 `git commit -F 文件`，写完先删再 `git add`
- 日志在 `run/logs/latest.log`（INFO 级），debug 级在 `debug.log`
- 服务器进程用 `Get-Process java | Stop-Process -Force` 杀

## 版本策略

主版本锁 **26.1.2**（旧版本玩家是基本盘）。26.3 噪声栈重构（float 化、区域采样 API）
将来多版本适配时对反编译源码核实，不凭 yarn 名猜。
