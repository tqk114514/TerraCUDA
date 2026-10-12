# TerraCUDA 项目接手指南

> 写给下一个会话的我。mod 的契约只有一句话：**NOISE 阶段在 GPU 上、与 vanilla 逐位一致，
> 其余十一个阶段一个字节不碰。** 这份指南是唯一需要读的东西——读完就能干活。

## 一句话定位

NeoForge mod（MIT），把 Minecraft 26.1.2 的 NOISE 阶段（地形密度 → 方块）卸载到 NVIDIA
GPU（CUDA），与 vanilla 逐位一致；其余阶段全部原样走 vanilla（包括它们的调度）。
MC 26.1.2 / NeoForge 26.1.2.114 / Java 25 / `mod_id=terracuda`。远端 `github.com/tqk114514/TerraCUDA`。

## 当前数字

- vanilla 无头基线：Chunky 22,801 区块 **5:10**（噪声中位 79/s）——配置无关，长期有效。
- **纯 NOISE 构建的无头成绩待补测**（性能笔记第十三节；上次要跑时 ffmpeg 吃了约 4 核，
  按测量纪律放弃）。第十至十二节里的 2:55→2:44 测于带调度层实验的构建，**不是本 mod 的
  形态，不要引用**。
- 无头协议下 mod 的绑定约束是 vanilla 自己的**串行 dispatcher**（~79/s，第八节源码确认）。
  它在范围外：动它就是调度层，是 C2ME 的主场——范围判决见性能笔记第十三节。

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

## 下一步（按性价比序）

1. **补测纯 NOISE 无头成绩**（协议见下），写进性能笔记第十三节——当前最大欠账。
2. NOISE 内部延迟：设备 2.9 ms + rules 4.8 ms + 写回 2.1 ms。吞吐被 dispatcher 顶住，
   但玩家侧延迟（噪声 p50）仍有空间；改一处测一处，别信推断。
3. 26.3 多版本适配：噪声栈 float 化、区域采样 API。对反编译源码核实，不凭 yarn 名猜；
   `NoiseSampler.fill` 可能是比 fillFromNoise mixin 干净一个量级的注入点。

## 关键命令

```powershell
# 构建 + 测试（91 个，全绿是底线）
./gradlew test

# 无头基准（优先用这个，不需要玩家）
# 前提：run/server.properties 开 RCON(enable-rcon=true, rcon.password=terracuda)，level-seed=123
#       run/mods 装 Chunky，run/eula.txt eula=true
Remove-Item -Recurse -Force run/world, run/chunky -ErrorAction SilentlyContinue
./gradlew runServer -Ptakeover -Ptiming   # 后台启动
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
