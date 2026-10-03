# Minecraft 区块生成 GPU 化规划

**目标版本**：Minecraft 26.1.2（已去混淆）/ NeoForge 26.1.x（`java_version=25`，`moddevgradle`）
**目标平台**：Windows 优先（`nvcuda.dll`），NVIDIA GPU 独占（CUDA）；AMD/Intel 不做
**交付形态**：NeoForge mod，Java FFM API + CUDA Driver API + 预编译 cubin
**GPU 化目标**：`ChunkStatus.NOISE` 阶段（`NoiseBasedChunkGenerator.fillFromNoise`）——地形密度函数求值、含水层、矿脉、默认方块填充，是整条生成管线中 FLOP 密度最高的阶段
**本文档的性质**：规划文档，不含实现代码；所有行为描述均以 26.1.2 反编译源码为准

---

## 0. 摘要

Minecraft 的地形"密度"不是一张体素图，而是一棵**密度函数 DAG**（约 35 种节点类型），在每次区块生成时由 Java 虚调用解释器逐点求值。26.1.2 的主世界每个区块约执行：

- **9,800 次全 DAG 求值**（8 个 `interpolated` 插值器 × 5×5×49 单元格角点），每次求值包含数十个噪声 octave 采样（`BlendedNoise` 单点 ≈ 40+ 次 `ImprovedNoise.noise()`）；
- **98,304 次逐方块求值**（`cache_all_in_cell` 的 `final_density + beardifier`，768 单元格 × 128 样本），每次是"读插值结果 + 走材质规则"的轻量 DAG；
- **含水层**：每区块约百余个 aquifer 网格单元 + 逐方块 `computeSubstance`；
- **初步地表**（`preliminary_surface_level`）：每列一次 `find_top_surface` 下行扫描，且含水层会读取**跨区块 ±3 section** 的列。

这个结构对 GPU 极其友好：采样点彼此独立、噪声参数在整个存档生命周期内是常量、计算是逐位可复现的整数/浮点混合运算。**设计核心不是"把密度数组搬上 GPU"，而是把 Marker 缓存体系（插值器、FlatCache、CacheAllInCell、Cache2D、CacheOnce）原样镜像到 GPU 内存布局上**——这决定了正确性和性能同时成立。

调用开销压到接近零的路径是成立的：预编译 cubin 常驻、`cuLaunchKernel` 每批一次、固定 pinned staging buffer、event 驱动完成通知。单区块总成本将由"GPU kernel 几十 µs + CPU 写块 replay"主导，**CPU 侧 `setBlockState` 逐方块写回会成为新瓶颈**，因此输出层设计为"直接写 section 内部 PalettedContainer"的批量回填，而不是逐方块 `setBlockState`。

---

## 1. 术语与坐标系

| 层 | 粒度 | 说明 |
|---|---|---|
| block | 1 | 方块坐标，世界坐标系 |
| quart | 4 block | `QuartPos`：群系/部分噪声的采样粒度（`fromBlock = >>2`，`toBlock = <<2`） |
| section | 16 block | `LevelChunkSection` / `SectionPos`：存储与光照单位 |
| cell | `cellWidth×cellHeight` | 噪声插值单元。主世界 `size_horizontal=1`→`cellWidth=QuartPos.toBlock(1)=4`，`size_vertical=2`→`cellHeight=QuartPos.toBlock(2)=8` |
| chunk | 16×16 | `ChunkAccess`/`ProtoChunk`，256 列 |

主世界 `NoiseSettings`：`minY=-64, height=384, size_horizontal=1, size_vertical=2` ⇒ 每 chunk **4×4×48 = 768 cell**，角点网格 **5×5×49 = 1225** 点，每 cell 内 **4×4×8 = 128** 方块采样（合计 98,304 = 16×16×384）。下界 `(0,128,1,2)`、末地 `(0,128,2,1)` 等参数不同，架构必须参数化，不得硬编码。

---

## 2. Vanilla 生成管线精确还原（26.1.2 源码）

### 2.1 阶段链与调度

`ChunkStatus`（`net.minecraft.world.level.chunk.status`）注册顺序：

```
EMPTY → STRUCTURE_STARTS → STRUCTURE_REFERENCES → BIOMES → NOISE → SURFACE
      → CARVERS → FEATURES → INITIALIZE_LIGHT → LIGHT → SPAWN → FULL
```

`ChunkStatus` 本身只是"状态 + parent + heightmapsAfter + chunkType"，task 绑定在 `ChunkPyramid` 中（`GENERATION_PYRAMID`/`LOADING_PYRAMID`），例如 `ChunkPyramid` 内 `.setTask(ChunkStatusTasks::generateNoise)` 绑定 NOISE 步。

调度链（`server.level` 包）：

1. `ChunkMap.scheduleChunkGenerationTask` → `GenerationChunkHolder` → `ChunkGenerationTask`；
2. `ChunkGenerationTask.runUntilWait`：`scheduleNextLayer` 按 `ChunkPyramid` 累积半径对域内每个 chunk 调 `scheduleLayer` → `chunkHolder.applyStep(pyramid.getStepTo(status), chunkMap, cache)`；
3. `ChunkMap.runGenerationTask` 把 task 丢进 `ChunkTaskDispatcher`（`ChunkTaskPriorityQueue` + `TaskScheduler` + `PriorityConsecutiveExecutor(4, dispatcherExecutor)`），**按 ticket level 排序**——离玩家近的 chunk 先跑；
4. task 返回 `CompletableFuture` 时挂到 `scheduledLayer`，`future.thenRun(runGenerationTask)` 恢复执行。

关键事实：`fillFromNoise` 内部是 `CompletableFuture.supplyAsync(doFill, Util.backgroundExecutor().forName("wgen_fill_noise"))`。`Util.backgroundExecutor()` 是 `ForkJoinPool`，线程数 `clamp(availableProcessors-1, 1, maxThreads)`（默认上限 `max.bg.threads` 属性，≤255）。**vanilla 已经用几乎全部 CPU 核并行生成区块**——GPU 化要赢的不仅是单核性能，而是把这个池从"N-1 核 × 每区块数十 ms"变成"排队 + 批量 offload"。

`generateNoise`（`ChunkStatusTasks`）本体：

```java
context.generator().fillFromNoise(
    Blender.of(region),
    level.getChunkSource().randomState(),
    level.structureManager().forWorldGenRegion(region),
    chunk
).thenApply(c -> /* BelowZeroRetrogen bedrock 处理，若有 */);
```

### 2.2 NOISE 阶段内部结构

`NoiseBasedChunkGenerator.fillFromNoise(Blender, RandomState, StructureManager, ChunkAccess)`（26.1.2 无 Executor 参数）：

1. `createNoiseChunk`：`NoiseChunk.forChunk(chunk, randomState, Beardifier.forStructuresInChunk(structureManager, chunk.getPos()), settings.value(), globalFluidPicker.get(), blender)`
2. `globalFluidPicker = (x,y,z) -> y < Math.min(-54, seaLevel) ? LAVA_FLUID : SEA_FLUID`
3. `doFill`：
   - 自顶向下获取 `LevelChunkSection`（每 cell 高 8 = 半 section，与 16-block section 边界对齐）；
   - `noiseChunk.initializeForFirstCellX()`：填第一个 cellX 的 5 个 slice；
   - 循环 `cellXIndex(0..3) → advanceCellX` → `cellZIndex(0..3)` → `cellYIndex`（自上而下）→ `selectCellYZ`：
     - 每个 `NoiseInterpolator.selectCellYZ` 绑好 8 角；
     - 每个 `CacheAllInCell`（只有 `fullNoiseDensity` 一个）调 `fillAllDirectly`：按 `yInCell desc → xInCell → zInCell` 填 128 个值——**每个值是一次完整 `final_density + beardifier` 求值**，其中 `interpolated` 叶子读插值结果；
   - 对 cell 内 128 个方块：`noiseChunk.getInterpolatedState()` → `blockStateRule`（`MaterialRuleList`：先 `aquifer.computeSubstance`，再 `OreVeinifier`）→ null 则 `settings.defaultBlock()`；
   - `section.setBlockState(x,y,z,state,false)` + `oceanFloor.update` + `worldSurface.update`（OCEAN_FLOOR_WG、WORLD_SURFACE_WG）+ `aquifer.shouldScheduleFluidUpdate() && !state.getFluidState().isEmpty()` → `markPosForPostprocessing`；
   - `swapSlices()`、`stopInterpolation()`。
4. `createBiomes`（BIOMES 阶段，NOISE 之前）：`protoChunk.fillBiomesFromNoise(biomeResolver, noiseChunk.cachedClimateSampler(router, spawnTarget))`——quart 粒度，便宜。
5. `applyCarvers`（CARVERS）：使用 `noiseChunk.aquifer()`，17×17 区块邻域。
6. `buildSurface`（SURFACE）：`randomState.surfaceSystem().buildSurface(...)`，`SurfaceRules.Context` 会回读 `noiseChunk.preliminarySurfaceLevel`（ quart 粒度缓存）——**NOISE 产物对 SURFACE 阶段有依赖**。

### 2.3 DensityFunction DAG 与 Marker 缓存体系

`DensityFunctions` 注册的节点类型（全集，mod 的 DAG 编译器必须覆盖或 fail-soft）：

- **叶子**：`constant`、`noise`（NormalNoise+scale）、`old_blended_noise`（BlendedNoise）、`shift_a/shift_b`、`shift`（offset noise）、`shifted_noise`、`end_islands`、`blend_alpha`、`blend_offset`、`blend_density`、`beardifier`、`y`（y_clamped_gradient 特殊通道）、`find_top_surface` 的 upperBound 等；
- **算子**：`add/mul/min/max`（含 MulOrAdd 常量折叠）、`abs/square/cube/half_negative/quarter_negative/invert/squeeze`、`clamp`、`lerp`、`blend_density`、`range_choice`、`y_clamped_gradient`、`weird_scaled_sampler`（RarityValueMapper TYPE1/TYPE2）、`spline`（CubicSpline，多维嵌套）、`find_top_surface`；
- **Marker**（`DensityFunctions.Marker`，`NoiseChunk.wrapNew` 映射为缓存对象）：

| Marker | 语义（源码） | GPU 对应 |
|---|---|---|
| `interpolated` → `NoiseInterpolator` | 函数只在 cell **角点**（5×5×49）求值；块内读 `lerp3` 三线性插值，顺序 Y→X→Z | **Kernel 2**：每角点一次完整子 DAG 求值写 `float[8 插值器][5][5][49]`；块内读取退化为 8 次 lerp |
| `flat_cache` → `FlatCache` | 5×5 quart 网格列惰性填充（`sizeXZ = noiseSizeXZ+1`），按 `QuartPos.fromBlock` 索引 | **Kernel 1**：每列一次求值写 `float[N][5][5]` |
| `cache_2d` → `Cache2D` | 按精确 (blockX,blockZ) 单值 memoize | 2D 表或"同列内 DAG 子表达式复用" |
| `cache_once` → `CacheOnce` | 同一次采样上下文内单值缓存；fillArray 复用上次数组 | cell kernel 内 shared/reg 复用 |
| `cache_all_in_cell` → `CacheAllInCell` | 每 cell 128 值数组（`final_density` 专用） | **Kernel 3** 的输出本体 |

overworld `final_density` 结构（`noise_settings/overworld.json`）：

```
final_density = min(
    squeeze(0.64 * interpolated(               // ← 插值器 #1，角点级
        blend_density(                          // Blender：无 blend 数据时为恒等
            0.1171875
            + ygrad(-64→-40 : 0→1)
              * ( -0.1171875 + (-0.078125
                  + ygrad(240→256 : 1→0)
                    * (0.078125 + range_choice(sloped_cheese,
                        in(-1e6,1.5625)  → min(sloped_cheese, 5*entrances),
                        out              → max( min( min(4*square(cave_layer)
                                                     + clamp(0.27+cave_cheese,-1,1),
                                                     clamp(1.5-0.64*sloped_cheese,0,0.5)),
                                                 entrances),
                                             spaghetti_2d + spaghetti_roughness),
                                         range_choice(pillars <0.03 → -1e6))
                )))
        )
    ),
    noodle                                      // ← min() 外层参数，内部再含 4 个 interpolated
)
```

其中 `sloped_cheese = 4 * quarterNegative((depth + jaggedness * halfNegative(jagged噪声)) * factor) + base_3d_noise`；`base_3d_noise` 是 `old_blended_noise`（BlendedNoise：3 个 legacy Perlin，每点约 40 次 ImprovedNoise 采样，**单点最贵叶子**）。`vein_toggle`/`vein_ridged`(×2)/`noodle`（×4）再贡献 7 个 `interpolated`，合计 **8 个插值器**。

`preliminary_surface_level` = `find_top_surface(cellHeight=8)`：`upperBound = clamp(128 - 128*(0.2734375/factor - offset), -40, 320)`，从 topY 按 8 向下扫 `density > 0`，密度链**只含 cache_2d(offset/factor) + 梯度 + clamp，无 3D 噪声**——每列几十次廉价评估。

### 2.4 噪声与随机数体系（逐位一致所必需）

- **NormalNoise** = `(first.getValue(x,y,z) + second.getValue(x*1.0181268882175227, …)) * (0.1666…/expectedDeviation)`；`expectedDeviation = 0.1*(1+1/(octaveSpan+1))`。
- **PerlinNoise**：`noiseLevels[i] = ImprovedNoise(positional.fromHashOf("octave_"+i))`（跳过 amplitude=0 的 octave）；采样 `factor=2^(-firstOctave)` 起步、`valueFactor=2^(octaves-1)/(2^octaves-1)` 起步逐 octave 翻倍/减半；坐标 `wrap(x) = x - floor(x/33554432 + 0.5)*33554432`（double 运算，必须原样）。
- **ImprovedNoise**：`p[256]` 置换表（Fisher-Yates `nextInt(256-i)` 交换）+ `xo/yo/zo ∈ [0,256)`；`noise(x,y,z,yScale,yFudge)` = 8 角 `gradDot(p[…]&15 → GRADIENT 表)` 经 `smoothstep` lerp3；`yrFudge` 分支有 `1e-7f` 浮点 fudge。
- **XoroshiroRandomSource**：`Xoroshiro128PlusPlus`（`rotl(s0+s1,17)+s0` 等，纯 64-bit 位运算）；`forkPositional()` 消耗两抽 `nextLong` 得 `(seedLo,seedHi)`；`at(x,y,z)` = `new Xoroshiro(seedLo ^ Mth.getSeed(x,y,z), seedHi)`；`Mth.getSeed` = `x*3129871 ^ z*116129781 ^ y; seed = seed*seed*42317861 + seed*11; return seed>>16`；`fromHashOf(name)` = MD5(name) 前/后 8 字节拼两个 long 再 xor；`nextInt(bound)` 用 `__umulhi` 等价的无符号高位乘 + 拒绝采样循环——**全部可位级移植到 CUDA**。
- **关键设计决策**：**不要在 GPU/宿主侧重新实现种子推导**。mod 运行在 MC runtime 内，`RandomState` 由 vanilla 代码照常构建；初始化时用反射/Accessor 直接从 vanilla 对象导出每个 `NormalNoise` 的 `ImprovedNoise.p[256]`、`xo/yo/zo`、`firstOctave/amplitudes`，以及各 `positional factory` 的 `(seedLo,seedHi)`，打包上传设备。这样无论数据包/其他 mod 怎么改噪声参数，继承的都是同一套已实例化噪声——seeding 路径零分叉。

### 2.5 含水层（Aquifer）

`NoiseBasedAquifer`（每 chunk 一个）：

- 网格：`gridX = x>>4`（16 块间距），y 向 12 块间距；单元内位置用 `aquiferRandom.at(gx,gy,gz)` 随机化（`nextInt(10/9/10)` 偏移）；`minGridX = gridX(minX-5)` 等带边界；
- `aquiferCache`（LazyFluidStatus[]）、`aquiferLocationCache`（BlockPos[]）懒初始化；
- `skipSamplingAboveY` = 本 chunk 各 quart 列 `preliminarySurfaceLevel` 最大值（+12 调整）；
- `computeSubstance(ctx, density)`：`density>0`→null；`y>skipSamplingAboveY`→`globalFluidPicker`；全局流体是岩浆→岩浆；否则在 2×3×2 网格邻域取 **4 个最近 aquifer 位置**，`similarity(d1,d2)=1-(d2-d1)/25`，`calculatePressure`（barrierNoise + 液面差 + 分段梯度 ±2.0）+ `mayFlow` → 返回液体 `BlockState` 或 null；置 `shouldScheduleFluidUpdate`；
- `computeFluid`：沿 `SURFACE_SAMPLING_OFFSETS_IN_CHUNKS`（13 个偏移，**±3 section**）扫邻列 `preliminarySurfaceLevel`（`+8` 调整），`OverworldBiomeBuilder.isDeepDarkRegion(erosion<-0.225 && depth>0.9)` → 深暗区特殊处理；否则 floodedness 噪声 clamp + 距地表衰减（`clampedMap(distBelowSurface,0,64,1,0)`）；
- `computeRandomizedFluidSurfaceLevel`：16×40×16 液面 cell，`fluidLevelSpreadNoise*10` 后 `Mth.quantize(…,3)`；
- `computeFluidType`：`|lavaNoise|>0.3` 且液面 ≤-10 → 岩浆。

**跨区块依赖就在这一层**：含水层读取自身边界外 ±3 section 的初步地表——批处理的域必须按 halo 扩张（见 §5.4）。

### 2.6 其余逐块规则

- **OreVeinifier**：每方块 `oreRandom.at(x,y,z)` 现场派生 xoroshiro（`nextFloat>0.7`→空、`richness` 门限、`0.02` 粗矿块）；铜脉 y∈[0,50]、铁脉 y∈[-60,-8]；读 `vein_toggle`/`vein_ridged`/`vein_gap`（前两者含 interpolated）。
- **Beardifier**：`forStructuresInChunk` 产 `Rigid(BoundingBox, TerrainAdjustment, groundLevelDelta)` + junctions（±12 邻域）。`compute` 是分段贡献：BURY→`clampedMap(len,0,6)`；BEARD_*→`-dy*rsqrt(dist²/2)/2 * BEARD_KERNEL[idx]`（24³ `exp(-d²/16)` 预烤表，索引 +12 偏移）；ENCAPSULATE→缩小版 bury；junction 权重 0.4，rigid 0.8。结构数据是 CPU 对象——**按 chunk 序列化小型数组随批次上传**。
- **Blender**：旧世界过渡。`blend_alpha/blend_offset` FlatCache，`blend_density` 对 `interpolated` 内部整体 lerp 新旧密度。带 blend 数据的区域极少——**直接走 CPU fallback**（见 §7）。
- **Heightmap**：`OCEAN_FLOOR_WG`/`WORLD_SURFACE_WG` 在写块时更新；`isOpaque` 谓词按 type 不同（`WORLD_SURFACE_WG` 判定非空气、`OCEAN_FLOOR` 判定 `blocksMotion`……实现时逐 type 抄谓词）。
- **markPosForPostprocessing**：`ProtoChunk.postProcessing` per-section `ShortList`，`packOffsetCoordinates = dx | dy<<4 | dz<<8`。

### 2.7 单 chunk 工作量定量模型

| 项 | 每 chunk 数量 | 单次成本 |
|---|---|---|
| `interpolated` 角点求值 | 8 × 5×5×49 = **9,800** | 重：BlendedNoise≈40+ ImprovedNoise；普通 noise 1-9 octave×2 perlin |
| `cache_all_in_cell` 求值 | 768 × 128 = **98,304** | 轻：lerp 读 + DAG 走查 + aquifer+ore 规则 |
| `flat_cache`/`cache_2d` 列求值 | ~5×5×N + prelim 扫描 ≤46 步/列 | 中：2D spline + shifted_noise |
| aquifer 网格/邻域 | ~百余 grid cell + 13-offset 表面扫描 | 中 |
| 高度图/后处理标记 | 98,304 次谓词 | 微 |

---

## 3. 总体架构

```
┌─ NeoForge mod (Java 25) ────────────────────────────────────────────┐
│ Mixin: NoiseBasedChunkGenerator#fillFromNoise  (HEAD, cancellable)   │
│   └→ GpuChunkService.submit(JobSpec) → CompletableFuture<ChunkAccess>│
│                                                                      │
│ GpuDispatcherThread (独占 CUDA context)                              │
│   └→ BatchBuilder: 同 (worldSeed → RandomState id + NoiseSettings)   │
│      的 chunk 按到达序聚合；阈值 maxBatch 或超时 flush                │
│   └→ H2D(pinned) → K1..K5 → D2H(pinned) → cuEvent                    │
│                                                                      │
│ CompletionThread: event sync → 分发结果 → WorkerReplay 写 section     │
│   └→ future.complete(chunk)                                          │
├─ Native 层 ─────────────────────────────────────────────────────────┤
│ FFM: Linker.downcallHandle → nvcuda.dll (Driver API)                 │
│ 常驻: CUcontext + CUlibrary(预编译 cubin) + pinned staging + 2 stream │
├─ CUDA 层（cubin，C/CUDA 编写）───────────────────────────────────────┤
│ K0 prelim_surface(halo rect)    K1 flat_cache/cache_2d 列            │
│ K2 interpolator 角点            K3 cell 内 128 采样 + 材质规则        │
│ K4 emit: blockId + heightmap + postprocess 位图                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.1 Java 侧组件

- **Mixin 接入点**：`NoiseBasedChunkGenerator.fillFromNoise`（`@Inject(method="fillFromNoise", at=@At("HEAD"), cancellable=true)`）。前置条件未满足（非主世界路由、blender 非空、GPU 未就绪、自定义 ChunkGenerator）→ 不 cancel，走 vanilla。
- **JobSpec**（入队时构造，全是不可变 POD）：`chunkPos, minY/height/cellW/cellH, router/dag 句柄 id, RandomStateRef（world 级）, beardifier payload[], globalFluidPicker(seaLevel, minY), blenderEmpty?`。
- **GpuChunkService**：per-world 生命周期，挂 `ServerLevel`/`MinecraftServer` 事件；持有 `GpuContext`（一次 `cuInit`+`cuCtxCreate`，常驻）。
- **配置**：`enabled, maxBatch, batchTimeoutMs, deviceId, fallbackToCpu, debugParityMode`。

### 3.2 kernel 策略选型：解释器 vs 生成式

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A. 通用 DAG 解释器 | 运行期把 DAG flatten 为指令数组（op + 立即数 + 子节点索引），GPU 线程按栈式/寄存器分配求值 | 兼容任意数据包/modded noise_settings | 每点有解释开销；寄存器压力大 |
| B. 离线特化 kernel | 把 vanilla overworld DAG 代码生成为 CUDA C，`-cubin` 编译 | 最快、寄存器可控 | 只对固定 DAG；数据包一改就废 |
| **C. 混合（推荐）** | 实现 A 作为通用路径；对"逐节点结构等于 vanilla overworld DAG"的会话启用 B 的特化 kernel | 正确性与速度兼得；A 也是 B 的 parity 参考 | 两套实现要维护 |

DAG lowering 规则：Marker 节点不展开进指令流——它们是 kernel 边界；子 DAG 顶点集合 = {const, noise, mapped, add/mul/min/max, spline, range_choice, ygrad, shift_a/b, shift, shifted_noise, weird_scaled, blend_*, beardifier, find_top_surface, old_blended_noise}。`HolderHolder`（跨文件引用如 `overworld/sloped_cheese`）在 lowering 时内联展开并按对象 identity 去重。

### 3.3 每世界初始化（GPU context 填充时序）

`RandomState.create` 完成后（`ChunkMap` 构造或 `ServerLevel` 加载钩子）：

1. walk `randomState.router()` + noise_settings DAG，收集所有 `NormalNoise`/`BlendedNoise`/`SimplexNoise` 实例 → 反射导出 `ImprovedNoise.p[256]`（`byte`）、`xo/yo/zo`（`double`）、`noiseLevels` 非空位图、`amplitudes`；
2. 导出 positional factories：`(seedLo,seedHi)` of `aquiferRandom`/`oreRandom`/`random.fromHashOf("offset")` 等——直接读对象字段；
3. 扁平化 spline：CubicSpline 递归为"坐标索引 + 断点数组 + 导数"结构体；
4. 打包上传为一个只读 `__constant__`/global blob + 每噪声 desc 表；同 world 多 chunk 零成本共享。

### 3.4 kernel 阶段划分（单 batch）

设 batch 内 chunk 集合 C，bounding rect R = ∪C + halo（**四周各 +3 section = 48 块**，覆盖 aquifer 表面采样域；Beardifier 影响仅本 chunk）。

| Kernel | 网格 | 工作 |
|---|---|---|
| K0 preliminary_surface | R 的 quart 列（(16N+96)/4 ≈ 每 chunk 摊到 ~36 列） | `find_top_surface`：upperBound 求值 → 按 cellHeight 下行扫 density>0；输出 `int prelim[col]` |
| K1 flat_2d | 每 chunk 5×5 quart 网格 × 每 FlatCache/Cache2D 节点 | shifted_noise、spline 链；输出 per-node 2D 表 |
| K2 interpolator_corners | 每 chunk 5×5×49 角点 × 8 插值器 | 完整子 DAG（最重 kernel）；输出 `float corners[interp][25][49]` |
| K3 cell_fill | 每 chunk 768 cell × 128 线程 | 读插值表做 lerp3(Y→X→Z) → `final_density+beard` → MaterialRule（aquifer→veinifier）→ `u16 blockId` + flag 位 |
| K4 emit | 每列/每方块 | 谓词 → heightmap 原子最小值；postprocess 位图（98304 bit/chunk） |

aquifer 嵌入 K3（per-block）+ K0/K1 提供查表；aquifer 网格位置（aquiferRandom.at）在 K3 内即时派生 xoroshiro——每块仅 ~15 条位运算。

### 3.5 输出格式与 CPU replay

D2H 返回 per chunk：

- `u16 blockStateId[16][16][384]`（196 KB；palette 为 `defaultBlock + 流体 + 矿脉方块` 的小表——严格说是 `u8` 也够，留 u16 兼容 modded 方块）；
- `u8 flags[…]` 位图：bit0 = aquifer 后处理；
- `i32 heightmap[2][256]`（OCEAN_FLOOR_WG、WORLD_SURFACE_WG 直接输出终值）；
- `i32 prelimSurface` halo 表（供 SURFACE 阶段 `SurfaceRules.Context` 复用——把结果灌进 `noiseChunk.preliminarySurfaceLevelCache`，SURFACE 走 vanilla）。

**CPU replay 不做逐块 `setBlockState`**：直接构造 `LevelChunkSection` 内部 `PalettedContainer`/`DataLayer`（section 是 16³=4096 槽位；98304 = 24 sections × 4096）。palatte 生成策略：先一遍数直方图（CPU 上 98K u16 → 通常 <16 种 → single-value/linear palette），再直写 `PalettedContainer` 数据。此步必须做——`setBlockState` 逐方块走锁+谓词+heightmap 在 98K 次调用下比 GPU 计算还贵。

### 3.6 FFM + CUDA Driver API 集成

- **加载**：`SymbolLookup.libraryLookup("nvcuda", arena)`（Windows 自动解析 `nvcuda.dll`）→ `Linker.nativeLinker().downcallHandle(symbol, FunctionDescriptor)`。JDK 25 FFM 全部稳定；高频 handle 加 `Linker.Option.critical()`（禁用堆内拷贝回填）进一步压开销。
- **函数集**（Driver API，`FunctionDescriptor` 的 C 类型注意 `long`/`size_t` 位宽）：
  `cuInit(0)`、`cuDeviceGet`、`cuDeviceGetAttribute`、`cuDeviceGetName`、`cuCtxCreate_v2`（或 `cuCtxSetCurrent`）、`cuLibraryLoadData`/`cuModuleLoadData`（PTX 或 cubin bytes）、`cuLibraryGetKernel`/`cuModuleGetFunction`、`cuKernelSetAttribute`、`cuLaunchKernel`、`cuMemAlloc_v2`/`cuMemFree_v2`、`cuMemHostAlloc`/`cuMemHostRegister_v2`（**把 FFM 分配的 `MemorySegment` 注册为 pinned**——pageable 的 `MemcpyAsync` 会退回同步）、`cuMemcpyHtoDAsync_v2`/`DtoHAsync_v2`、`cuStreamCreate`、`cuEventCreate/Record/Synchronize/ElapsedTime`、`cuStreamWaitEvent`。
- **错误处理**：每个 CUresult≠0 → `cuGetErrorName` 解析 → `GpuWorldgenException` → 该 world 标记 degraded，后续 chunk 全部走 vanilla（fail-soft，永不 crash JVM）。
- **cubin 工具链**：构建期 `nvcc -cubin -arch=sm_XX`（矩阵：sm_75/80/86/89/90/100/120 覆盖 RTX 20-50 系）打包进 mod jar `META-INF/cubin/`；runtime `cuDeviceGetAttribute(COMPUTE_CAPABILITY_*)` 选最近 cubin，找不到 → 加载嵌入 PTX 由驱动 JIT，再不行 → CPU fallback。
- **调用开销预算**（验证"几乎为零"）：

| 操作 | 量级 |
|---|---|
| `downcallHandle.invoke` per call | ~50-100 ns（critical 模式更低） |
| `cuLaunchKernel` | ~3-5 µs |
| H2D/D2H async（pinned，PCIe4 x16 ≈ 25 GB/s） | 每批 ~2 MB ≈ 80 µs，可与计算重叠 |
| 事件同步 | `cuEventSynchronize` 轮询 ~2 µs |
| **每 batch 固定开销** | **~10-20 µs**，摊到 8-16 chunk ≈ **1-2 µs/chunk** |

---

## 4. 批处理设计（含参考资料评审）

### 4.1 修正后的接入语义

- 接入点正确性是参考资料里少数站得住的部分：返回 `CompletableFuture` 确实是攒批的天然抓手。**但 26.1.2 里方法叫 `fillFromNoise`，签名是 `(Blender, RandomState, StructureManager, ChunkAccess)`——无 Executor 参数**（内部 `supplyAsync` 到 `Util.backgroundExecutor`），返回 `CompletableFuture<ChunkAccess>`。
- 上游 `ChunkTaskDispatcher` 已按 ticket level 排序到达——**批队列维持 FIFO 即可，不重排**（重排会饿死远处 chunk 且违背 ticket 语义）。
- `Blender.of(region)` 按域构造：入队时判定 `blender.isEmpty()`，非空 chunk 标 CPU-only 混入同一 future 管道（见 §7）。

### 4.2 批策略

- **阈值**：`maxBatch=16`，`batchTimeoutMs=3`；低负载（队列<4 且>0）时 1ms 快速通道——**GPU 空转成本几乎为零，延迟权重高于吞吐**。
- **单批耗时上界**：不按"tick 50ms"拍脑袋（chunk gen 本来异步，不占 tick），而按"future 未决时延影响下游 chunk 的 BIOMES/SURFACE 链"——经验上界 ~25ms/批。超过则切分。
- **双缓冲**：pinned staging A/B + 2 stream——stream0 执行 batch_i 的 kernel，stream1 同时把 batch_{i+1} H2D。PCIe 传输与计算交叠后 H2D 成本归零。
- **halo**：K0 的 prelim 域 = batch bounding rect + 48 块边距。批内任意 chunk 的 aquifer 采样都只读 K0 输出，不需要邻居 chunk 的成品密度——**依赖被解耦成"只读公共 2D 表"**，这是批处理在 GPU 上成立的真正原因，参考资料完全没意识到这一点。

### 4.3 对用户参考资料的逐条评审

| # | 参考资料论点 | 判定 | 说明 |
|---|---|---|---|
| 1 | `populateNoise(Executor,…)` 签名 + future 接入 | **部分过时** | 26.1.2 为 `fillFromNoise(Blender, RandomState, StructureManager, ChunkAccess)`，无 Executor；future 接入点结论仍成立 |
| 2 | "密度数组 16×16×384/区块" | **模型错误** | vanilla 不存在逐体素密度数组；是 5×5×49 角点插值 + cell 内 128 次逐块求值的缓存结构。按 flat 体素做会**多算 ~8 倍且结果不等价**（插值器内含全部洞穴链，逐体素算出的洞穴边界更锐利——不是 vanilla 地形） |
| 3 | 批=收集/调度/回填三层 + 阈值+超时 | **成立** | 与本方案 §4.2 一致 |
| 4 | "双缓冲 stream 摊掉 PCIe" | **成立但缺前提** | 必须 pinned host 内存（`cuMemHostRegister`/`cuMemHostAlloc`）才能真 async；资料没提 |
| 5 | "16 chunk × 400KB = 6.4MB" | **数量级错** | 输出应取 blockId(u16)=196KB/chunk + 小表，而非 double 密度；16 chunk ≈ 3.2MB |
| 6 | "4090 上 16 chunk ≈ 1-2ms" | **量级可信但无依据** | 需实测；瓶颈大概率在 CPU replay 写块而非 GPU |
| 7 | "tick 期望 50ms 内完成" | **归因错误** | chunk gen 异步于 server tick；真约束是 future 时延×下游阶段串行 + 玩家体感 |
| 8 | "预测性预取再扩 1-2 圈" | **弱** | 未入队 chunk 没有 `ProtoChunk`/`GenerationChunkHolder` 承载结果；只能缓存密度贴图，而密度只是中间量。建议仅对**已进入 ticket 队列**的 chunk 做窗口合并，预取作为可选增强且默认关 |
| 9 | "后续 carve/buildSurface 依赖邻居，批处理管不了" | **部分对** | 关键是 aquifer 的 ±3 section prelim 读取——已由 halo 设计解决；FEATURES 邻居依赖在 vanilla 逻辑外，不归本 mod 管 |
| 10 | "Paper/Folia 服务器" | **无关** | 本 mod 是 NeoForge，integrated/dedicated server 两态即可 |
| 11 | "把多区块拼成 64×64×384 3D 纹理" | **可用但不必 texture** | linear buffer + 行主序 pitch 更灵活；真正要注意的是 halo 与 cell 对齐（区块间 cell 网格连续，反而有利于批内统一索引） |

**结论**：参考资料的方向（future 攒批 + 双缓冲 + 超时）是对的，但计算模型、数据规模、约束归因都有实质错误，按它直接实现会得到"更快但地形不对"的 mod。

---

## 5. 正确性约束清单（逐条对照源码）

1. **浮点语义**：Java 17+ 默认 strict；CUDA 用 `-fmad=false` 且禁 `--use_fast_math`；`ImprovedNoise`/`PerlinNoise`/`NormalNoise`/全部 DAG 节点用 **double**（消费卡 FP64 吞吐低，但每 chunk ~5-10 MFLOP × FP64 ≈ 4090 上仍 <100µs，可接受；若实测瓶颈可再评估"插值器内 double、cell fill 内 float"的混合精度并给出 parity 报告）。
2. **位运算**：xoroshiro 系列、`Mth.getSeed`、`nextInt(bound)` 的 `uint×bound` 高位乘（CUDA `__umulhi`/`__umul64hi`）+ 拒绝采样循环、MD5(name) 种子派生——全部 64/32 位逐位一致。
3. **采样网格对齐**：quart 坐标先 `QuartPos.fromBlock` 再 `toBlock`（`>>2<<2`）；插值 lerp3 顺序 **Y→X→Z**（`lerp(fx, lerp(fy,n000,n010), lerp(fy,n100,n110))` → `lerp(fz,…)`）；`forIndex`/`fillAllDirectly` 的 `y desc→x→z` 线性序决定 `CacheAllInCell.values` 下标 `((cellH-1-y)*W+x)*W+z`。
4. **lazy→eager 等价**：`CacheOnce`/`Cache2D`/`FlatCache`/`aquiferCache`/`preliminarySurfaceLevelCache` 均为 memoization——GPU 预计算整张表与 vanilla 懒计算语义等价，前提是表的覆盖域 ⊇ vanilla 实际会访问的点集（halo 设计保证）。
5. **顺序副作用**：`markPosForPostprocessing` 的 ShortList 顺序 = vanilla 的 y-desc x-asc z-asc 遍历序（chunk 内行为序无关，列表序仅影响 tick 顺序——保持一致以稳妥）；`Beardifier` rigids/junctions 顺序无关（加法交换律）但 double 加法顺序敏感——GPU 上按固定索引序累加。
6. **谓词**：`isOpaque`（heightmap）、`getFluidState().isEmpty()`、`Aquifer.shouldScheduleFluidUpdate`、`OreVeinifier` 的概率阈值（`|oreVeininess|+edgeRoundoff<0.4` skip、`nextFloat>0.7`、`veinRidged>=0`、`richness` 区间 `[0.1,0.3]`、`nextFloat<0.02` 粗矿、`veinGap>-0.3`）逐条照搬。
7. **全局流体规则**：`(x,y,z) -> y < min(-54, seaLevel) ? lava : seaFluid`；aquifer `density>0 → null` 短路、deep-dark 判定（`erosion<-0.225 && depth>0.9`）、`|lava|>0.3 && level<=-10`。
8. **`stopInterpolation`/`swapSlices`**：GPU 一次性算全部 5 个 cellX 的 slices 与 vanilla 滚动双缓冲等价（slice 内容只依赖 cellX 列坐标）。
9. **BelowZeroRetrogen**：`generateNoise` 的 `thenApply` 阶段对旧世界 chunk 做基岩替换——走 vanilla 路径，GPU 输出块状态后照常执行。
10. **modded 密度函数**：DAG lowering 遇到未知节点类型 → 该 chunk（或整个 world）标 unsupported → vanilla。NeoForge 的 `PieceBeardifierModifier`（ Beardifier patch）产生的自定义 rigids 仍走"序列化数组上传"通道。

---

## 6. 正确性验证计划

1. **三方 parity harness**（最重要资产，先于 GPU 写）：
   - CPU reference：mod 内实现"顺序、无缓存"的 DAG 求值器（慢但显然正确）；
   - 对照组：vanilla `fillFromNoise` 产物 chunk（同 seed、同坐标）；
   - 被测组：GPU 路径产物；
   - 比对：`ProtoChunk` 全量 NBT（`block_states` 数据层 + `Heightmaps` + `PostProcessing` + `biomes`），逐位相等为 pass。
2. **种子矩阵**：≥10 个 seed × 每 seed 200 chunk（含出生点、边界、海洋、深暗区、蘑菇岛）+ 至少 2 个含 blending 的旧版本升级存档（走 CPU 分支验证 fallback）。
3. **位级诊断模式**：`debugParity=true` 时 GPU 额外导出每 cell 的 `final_density` 原始值，与 CPU reference 逐值 diff（定位到 `(cell, block, node)`）。
4. **随机流审计**：GPU 重实现的 `xoroshiro.at(x,y,z).nextFloat()` 抽样 10⁶ 点与 Java 版比对；`nextInt(bound)` 拒绝采样分支压力测试（bound=10/9/10 正是 aquifer 偏移）。

---

## 7. 兼容性与降级矩阵

| 场景 | 路径 |
|---|---|
| 无 N 卡 / 驱动旧 / CUDA 初始化失败 | 全量 vanilla，日志一行 INFO |
| 非 `NoiseBasedChunkGenerator`（FlatLevelSource、调试世界、modded generator） | vanilla |
| noise_settings DAG 含未知节点（数据包/modded density function type） | 该 world 永久 vanilla（启动时 DAG lowering 探测一次） |
| `blender` 非空（旧世界升级边界区） | 该 chunk vanilla，同批其余照常 GPU |
| nether / caves 维度 | 同架构可支持（router 更简）；end 的 `end_islands`（25×25 simplex 扫描）需独立 kernel，v1 可先 fallback |
| 其他 mod hook `fillFromNoise` | Mixin priority 调整；检测返回值被改写即退让 |

---

## 8. 性能模型与测量计划

- **CPU 基线**：`spark`/`JFR` 采 `wgen_fill_noise` 任务耗时；典型主世界 chunk 在 8 核机上 ~30-80ms/chunk（worker 共享）。目标：GPU 路径 wall-clock ≤5ms/chunk（含 replay）。
- **预估**：K2（角点 DAG）是绝对热点——9800 × 平均 ~200 flop ≈ 2 GFLOP/chunk FP64，4090 FP64 ≈ 1.3 TFLOPS ⇒ ~1.5ms/chunk 上限；实际多数节点是 FP64 mul/add 链，良好 occupancy 下应 <1ms。批 16 → ~10-20ms 一批，吞吐 ~1000+ chunk/s。
- **瓶颈预警**：若 FP64 不达标，第一优化位是"interpolator 角点改 FP32、cell fill 保留 FP64"（角点 FP32 误差会被插值平滑，但仍会破坏逐位一致——作为 `highPerformance` 配置项而非默认）。
- **测量**：K 级 `cuEventElapsedTime` + Java 侧 `System.nanoTime` 分段（入队/等待/H2D/kernel/D2H/replay），`debugTiming` 开关。

---

## 9. 里程碑

| M | 内容 | 验收 |
|---|---|---|
| M0 | CPU 顺序求值器（DAG 解释器，无缓存）+ NBT parity harness | 同 seed 100 chunk 与 vanilla 逐位一致 |
| M1 | FFM 最小闭环：cuInit→加载嵌入 cubin→ImprovedNoise.noise 单点求值→D2H | 10⁶ 随机点 vs Java ImprovedNoise 位级一致 |
| M2 | K0+K1+K2：单 chunk 插值器全表 GPU 化，CPU 读回做 fillFromNoise 其余步骤 | parity pass；`wgen_fill_noise` 耗时显著下降 |
| M3 | K3+K4：材质规则 + emit + PalettedContainer 直写 replay | 全量 NBT parity pass |
| M4 | 批处理 + pinned 双缓冲 + halo 合并 | ≥8 chunk 批摊销；P95 时延报告 |
| M5 | 降级/fallback 全覆盖 + 配置 + nether/caves 扩展 + 打包发布 | 无卡机器上零崩溃；CI（Win+Linux build） |

## 10. 风险清单

- **位级一致性**：fma 收缩、Java/CUDA 常量舍入（如 `0.16666666666666666` 字面量一致即可）、`Mth.lerp` 的 `a+(b-a)*c` 形态——逐节点对照，宁可 verbose 不合并项；
- **replay 写块成本**：若 PalettedContainer 直写仍慢于预期，退化为"仅重写非空气 section + 单方块 section 特判"；
- **CUDA 驱动 API 版本**：`cuLibrary*`（CUDA 12+）与 `cuModule*`（全版本）双路径，优先 `cuModuleLoadData` 求兼容；
- **NeoForge 版本漂移**：26.x 未混淆不代表稳定——Mixin 用 `@Inject` 而非 `@Overwrite`，所有反射字段名启动时校验，失败即 fallback；
- **GPU 异常**：`CUDA_ERROR_LAUNCH_TIMEOUT`（Windows TDR，单次 kernel >2s 被杀）——批内拆分保证单 kernel <100ms 上限；
- **显存**：staging+表+buffers ≈ 每 batch <64MB，不做 VRAM 常驻压力；
- **多 world**：每 `ServerLevel` 一个 `GpuContext` 句柄表，CUDA context 全局一个。

## 附录 A：源码索引（26.1.2，net.minecraft）

| 关注点 | 类 | 关键成员 |
|---|---|---|
| 阶段链 | `world.level.chunk.status.ChunkStatus` | 注册顺序、WORLDGEN_HEIGHTMAPS |
| 任务绑定 | `server.level.ChunkPyramid` | `.setTask(ChunkStatusTasks::generateNoise)` |
| NOISE task | `world.level.chunk.status.ChunkStatusTasks` | `generateNoise → fillFromNoise(...).thenApply(BelowZeroRetrogen)` |
| 调度 | `server.level.ChunkMap` / `ChunkGenerationTask` / `ChunkTaskDispatcher` | `runUntilWait/scheduledLayer/applyStep`；`PriorityConsecutiveExecutor` |
| 生成器 | `world.level.levelgen.NoiseBasedChunkGenerator` | `fillFromNoise/createNoiseChunk/doFill/createBiomes/applyCarvers/buildSurface` |
| 噪声容器 | `world.level.levelgen.NoiseChunk` | `wrapNew` marker 映射、`interpolators/cellCaches`、`fillSlice/fillAllDirectly/forIndex/selectCellYZ`、`FlatCache/Cache2D/CacheOnce/CacheAllInCell/NoiseInterpolator`、`preliminarySurfaceLevel` |
| 密度函数 | `world.level.levelgen.DensityFunctions` | 35 节点类型、Marker.Type、BlendedNoise、ShiftedNoise、WeirdScaledSampler、FindTopSurface、EndIslandDensityFunction |
| 路由 | `NoiseRouter` record | barrier/floodedness/spread/lava/temperature/vegetation/continents/erosion/depth/ridges/preliminarySurfaceLevel/finalDensity/veinToggle/veinRidged/veinGap |
| 含水层 | `world.level.levelgen.Aquifer` | `NoiseBasedAquifer`：grid 16×12×16、`similarity/calculatePressure/mayFlow/computeFluid/computeFluidType`、SURFACE_SAMPLING_OFFSETS_IN_CHUNKS |
| 矿脉 | `OreVeinifier` | 铜/铁脉参数、`oreRandom.at` |
| 结构密度 | `Beardifier` | Rigid/junction、BEARD_KERNEL 24³ |
| 旧世界 | `Blender` / `BelowZeroRetrogen` | blend_alpha/blend_offset/blend_density |
| 随机 | `XoroshiroRandomSource`/`Xoroshiro128PlusPlus`/`RandomSupport` | `forkPositional/fromHashOf/at/getSeed/mixStafford13/upgradeSeedTo128bit` |
| 噪声 | `synth.NormalNoise/PerlinNoise/ImprovedNoise` | octave 循环、wrap(2^25)、p[256]、gradDot、lerp3 |
| 工具 | `Util` | `backgroundExecutor().forName("wgen_fill_noise")` |

## 附录 B：主世界噪声参数（`worldgen/noise/*.json`，节选关键项）

`jagged` firstOctave=-16 amps 全 1（16 octave，最耗）；`continentalness` -9/9 amps；`cave_cheese` -8/9 amps；`base_3d_noise` 实为 `old_blended_noise`（3×16 octave legacy Perlin）；`offset` -3/[1,1,1,0]；aquifer 系 4 个单 octave；矿脉系 4 个单 octave；spaghetti/noodle 系 8 个单 octave；群系 `temperature/vegetation` 为 shifted_noise。60 个 NoiseParameters 全表在 `client_data/data/minecraft/worldgen/noise/`。

## 附录 C：FFM FunctionDescriptor 速查

```java
Linker L = Linker.nativeLinker();
Arena A = Arena.ofShared();
SymbolLookup cuda = SymbolLookup.libraryLookup("nvcuda", A); // Windows
MethodHandle cuInit = L.downcallHandle(cuda.find("cuInit").orElseThrow(),
    FunctionDescriptor.of(JAVA_INT, JAVA_INT));
MethodHandle cuLaunchKernel = L.downcallHandle(cuda.find("cuLaunchKernel").orElseThrow(),
    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT×6, JAVA_INT×2, JAVA_INT, ADDRESS, ADDRESS),
    Linker.Option.critical());
// HtoD 前：cuMemHostRegister_v2 把 FFM MemorySegment 注册为 pinned
```
