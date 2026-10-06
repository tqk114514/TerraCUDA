# TerraCUDA 项目接手指南

> 写给下一个会话的我。上一个会话把项目从"NOISE 阶段 GPU 化"推到了"全管线优化、超过 C2ME"。
> 这份指南是唯一需要读的东西——读完就能干活。

## 一句话定位

NeoForge mod（MIT），把 Minecraft 26.1.2 的 NOISE 阶段卸载到 NVIDIA GPU（CUDA），与 vanilla 逐位一致。
MC 26.1.2 / NeoForge 26.1.2.114 / Java 25 / `mod_id=terracuda`。远端 `github.com/tqk114514/TerraCUDA`。

## 当前数字（2026-10-06 收官）

| | vanilla | TerraCUDA | C2ME 0.4.0 |
|---|---|---|---|
| Chunky 22,801 | 5:10 | **2:06** | 2:19 |
| 噪声中位 | 79/s | ~200/s | 179/s |
| 对 vanilla | 1.0× | **2.5×** | 2.2× |

## 必读文件

| 文件 | 内容 |
|---|---|
| [`.workbuddy-ai/memory/MEMORY.md`](.workbuddy-ai/memory/MEMORY.md) | **先读这个**——精简版长期记忆 |
| [`docs/性能笔记.md`](docs/性能笔记.md) | 所有测量与决定（第一~十八节），数字以此为准 |
| [`docs/GPU_区块生成规划.md`](docs/GPU_区块生成规划.md) | 原始规划文档 |
| [`.workbuddy-ai/memory/2026-10-04.md`](.workbuddy-ai/memory/2026-10-04.md) | SURFACE 重开知识 + vanilla 内部速查 + 坑清单 |
| [`.workbuddy-ai/memory/archive-2026-10-06/`](.workbuddy-ai/memory/archive-2026-10-06/) | 精简前的原始记忆备份 |

## 已完成的全部优化（不需要重做）

- M0–M4 全闭环（规划文档定义的里程碑）
- P0：计时仪器（噪声计数、GPU 线程窗口、节流器状态、延迟分位数）
- P1：拆线（设备线程产密度 → replay workers 消费）
- P2：上游解耦（`-Poffload` 下放所有阶段体除 LIGHT、`-Pinflight` 节流器实验）
- 多流+pin：三段设备 pass 重叠（页锁定是前提）
- Worker 扩容：2 条 rules workers，每 worker 一套解释器
- FEATURES 专用线程：无串行回退（gate 保留但不激活，单线程不需要）
- beardifier 收编：宿主侧逐方块加 beard，不再让路 vanilla
- invokeExact：热路径 FFM 调用
- Enable-Native-Access manifest

## 当前瓶颈（下一个会话要做的事）

**~200 chunks/s 被 vanilla 状态系统的依赖链（波纹推进）卡着。** 所有组件有余量
（设备 70%、rules 63%、features 86%），但状态系统一次只推进一个区块的阶段。

**下一杠杆：并行化调度层**——mixin `ChunkTaskDispatcher`/`ConsecutiveExecutor`，
让多线程同时做依赖检查和阶段提交。预期 200→233/s（FEATURES 天花板）→ 1:38。
参考 C2ME 的 FlowSched + fixes-threading。需要深层 mixin vanilla 内部。

## 关键命令

```powershell
# 构建 + 测试（92 个，全绿是底线）
./gradlew test

# 无头基准（优先用这个，不需要玩家）
# 前提：run/server.properties 开 RCON(enable-rcon=true, rcon.password=terracuda)，level-seed=123
Remove-Item -Recurse -Force run/world, run/chunky -ErrorAction SilentlyContinue
./gradlew runServer -Ptakeover -Ptiming -Poffload   # 后台启动
.\benchmark\chunky-rcon.ps1 -Command "chunky shape square"
.\benchmark\chunky-rcon.ps1 -Command "chunky center 0 0"
.\benchmark\chunky-rcon.ps1 -Command "chunky radius 1200"
.\benchmark\chunky-rcon.ps1 -Command "chunky start"
# 等完成（~2 分钟），从 run/logs/latest.log 提取 "Total time" 和 "noise stage" 行
.\benchmark\chunky-rcon.ps1 -Command "stop"

# 带玩家验证（runClient 需要手动操作）
./gradlew runClient -Ptakeover -Ptiming -Poffload
# 种子 123，视距 32，传送 (1000, 1000) 站定

# 其他开关
./gradlew runServer -PnoFeatures   # 关闭 FEATURES 下放（诊断用）
./gradlew runServer -PrulesWorkers=3  # 改 worker 数
# -Pinflight=8 提高节流器（实验用）
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
MIT 部分可读可参考，但一行不抄。

## 测量教训（违反会浪费时间）

- 同配置单场端到端带宽 **±18%**——跨场比较看分段（机制性），端到端必须复跑
- 天花板 A ≈ 天花板 B 时，"A 是瓶颈"未经拆 A 无法证实
- busy% 含队列等待 ≠ 设备占用
- 仪表必须能看见它声称测量的改变；新仪器先对已知量校准
- **过了 thread safety 测试 ≠ 没有 bug**（VanillaRandomExport 的 HashMap 竞态在双 worker
  下沉默了好几个会话才暴露）

## 关键教训（项目方法论）

- 诊断对两个假设给出**相同**数字时，先怀疑诊断
- 消融（顶成常量再计时）比采样推断有说服力
- 每个优化前先确认它瞄准的是绑定约束——优化不绑定的资源是零收益
- 负优化也是数据：六场"比快更快"实验里五场持平或更慢，每一场都排除了一个假说

## 工作流约定

- 提交信息承载"为什么"和"之前哪个结论错了"（英文正文，格式参照 git log）
- 性能数字进 `docs/性能笔记.md`，提交信息不重复数字
- pwsh 不支持 heredoc——提交信息用 `git commit -F 文件`，写完先删再 `git add`
- `.workbuddy-ai/` 不在 git 里，精简后的原件在 `archive-2026-10-06/`
- 日志在 `run/logs/latest.log`（INFO 级），debug 级在 `debug.log`
- 服务器进程用 `Get-Process java | Stop-Process -Force` 杀

## 版本策略

主版本锁 **26.1.2**（旧版本玩家是基本盘）。26.3 噪声栈重构（float 化、区域采样 API）
将来多版本适配时对反编译源码核实，不凭 yarn 名猜。`NoiseSampler.fill` 可能是比
fillFromNoise mixin 干净一个量级的 GPU 注入点。
