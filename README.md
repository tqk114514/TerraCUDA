# TerraCUDA

Offloads Minecraft's terrain noise generation from your CPU to your NVIDIA GPU.

## What it does

Minecraft builds a chunk in twelve stages. TerraCUDA replaces **one** of them — the `NOISE` stage,
where the terrain's shape is decided — and leaves the other eleven to vanilla.

Inside that stage, the split is deliberate:

**On the GPU**

- the density function: the `5 × 5 × 49` interpolator corner grid of every marker, the trilinear blend
  down to block resolution, and the few dozen instructions above the markers;
- the ore vein functions.

**On the CPU**

- the material rules — the aquifer and the ore veinifier. They are branch- and table-driven rather
  than floating-point heavy, and moving them would mean a per-thread copy of their grids and caches
  for very little;
- turning the result into block state ids, and writing the chunk back.

The result is the same terrain vanilla generates, block for block, checked against the game's own
classes rather than against a reimplementation of them.

It is off by default. When on, the `NOISE` stage runs on the GPU and vanilla's own is cancelled;
anything that goes wrong falls through to vanilla, which writes every block itself.

## Requirements

- Minecraft / NeoForge `26.1.2` (NeoForge `26.1.2.114`)
- Java 25
- An NVIDIA GPU with CUDA support

Windows only for now: the driver library is loaded by its Windows name (`nvcuda.dll`), so on Linux the
mod reports the device as unavailable and quietly uses vanilla. Adding `libcuda.so.1` as a second
candidate is a one-line change in `CudaDriver`.

## Building

```bash
./gradlew build
```

The compiled mod jar is written to `build/libs/`.

## Testing

```bash
./gradlew test
```

The suite asserts bit identity, not closeness: the reimplementations of vanilla's random and noise
primitives are compared against the Minecraft classes themselves, and the deobfuscated game is on the
test classpath, so parity is asserted without launching anything. The same goes for the device path —
the chunks it produces are compared block for block against chunks vanilla generated, over real
terrain. Hardware-dependent CUDA tests skip themselves on machines without an NVIDIA driver.

`ChunkPassProfileTest` is the exception: it prints where a chunk's time goes, per pass, asserts
nothing, and skips without a device. It is kept because a number is the only thing that has ever
settled an argument about this code.

## Status

| Milestone | State |
|---|---|
| M0 — CPU reference + parity | **done.** The overworld density function is lowered to an instruction program and evaluated bit-identically to vanilla, and every noise it uses round-trips through the export path. |
| M1 — FFM + a kernel | **done.** A million points through the CUDA `ImprovedNoise` match the Java reference bit for bit. |
| M2 — K0/K1/K2 | **done.** `preliminary_surface_level` and every per-chunk marker table are computed on the device, bit-identically to the CPU reference. |
| M3 — K3/K4 + chunk replay | **done.** The material rules are ported and reproduce vanilla's block for every one of 294912 blocks across three chunks; the chunk's blocks are produced end to end and written back, with the heightmaps and fluid-update flags `doFill` also maintains. |
| M4 — batching, pinned buffers, P95 | **settled, each on a measurement or a decision.** Batching was measured and declined, pinned buffers were dismissed by the same arithmetic, and the P95 latency report — the one item that was neither — is now built behind `-Ptiming`, which keeps per-chunk samples and reports percentiles rather than sums. See [`docs/性能笔记.md`](docs/性能笔记.md). |

## Trying it in a game

Both switches are off by default, and both need `-Dterracuda.gpu=true` underneath them.

To run the whole chain — lowering, upload, density, material rules, block ids — inside a real world
**without changing what the game generates**:

```
./gradlew runClient -Pshadow
```

The hook computes each chunk's blocks on the device path and logs how long that took; it never
cancels the vanilla call and never touches the chunk. A device that cannot be used degrades to a log
line.

To let it **build the chunks instead**:

```
./gradlew runClient -Ptakeover
```

Vanilla's `NOISE` stage is cancelled and the device's blocks are written into the chunk. A dimension
whose chunks are not the height the generator is configured for, a legacy-world blend, and anything
that fails mid-flight all fall through to vanilla.

Add `-Pverbose` to log every chunk rather than every 256th. (`-Pterracuda.shadow` works too, but the
Windows shell splits it at the dot, so the dotless form is the one to use there.)

For benchmarking, `-Ptiming` reports the device thread and the noise stage per window — it also
works on a vanilla run, which is how the baseline is measured — and `-Pinflight=N` raises vanilla's
player-ticket in-flight limit from 4, which is the experiment described in
[`docs/性能笔记.md`](docs/性能笔记.md).

## GPU kernels

The CUDA source lives in `src/main/cuda/` and is shipped inside the jar. At runtime the mod picks the
best available image, in this order:

1. a cubin prebuilt for the device's exact compute capability;
2. a prebuilt PTX image, JIT-compiled by the driver;
3. the embedded CUDA source, compiled at runtime with NVRTC.

Rungs 1 and 2 are produced by `nvcc` at build time, for the architectures in
`terracuda.cuda.archs` (default `75,80,86,89,90,100,120`). While iterating on a kernel, build only
your own architecture:

```bash
./gradlew build -Pterracuda.cuda.archs=86
```

On Windows `nvcc` additionally requires the MSVC host compiler (`cl.exe`). Without either, the build
still succeeds — the task is skipped and rung 3 takes over, so the CUDA Toolkit alone is enough to
run the GPU path; it just costs a one-off JIT at startup instead of none.

Whichever rung is used, the kernels are compiled with `--fmad=false` and never with
`--use_fast_math`. FMA contraction would make the GPU compute a differently-rounded answer than Java
does, and the terrain would diverge from vanilla.

## Design notes

- [`docs/GPU_区块生成规划.md`](docs/GPU_区块生成规划.md) — the implementation plan.
- [`docs/性能笔记.md`](docs/性能笔记.md) — the measurements, and the decisions they forced.

## Development

Open the project in IntelliJ IDEA or Eclipse. To refresh dependencies:

```bash
./gradlew --refresh-dependencies
```

To run a development client:

```bash
./gradlew runClient
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
