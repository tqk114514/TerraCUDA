# TerraCUDA

Offloads Minecraft's terrain noise generation from your CPU to your NVIDIA GPU.

## Requirements

- Minecraft / NeoForge `26.1.2` (NeoForge `26.1.2.114`)
- Java 25
- An NVIDIA GPU with CUDA support

## Building

```bash
./gradlew build
```

The compiled mod jar is written to `build/libs/`.

## Testing

```bash
./gradlew test
```

The test suite checks the bit-exact reimplementations of vanilla's random and noise primitives
(`Xoroshiro128PlusPlus`, `RandomSupport`, `Mth`, `ImprovedNoise`, `PerlinNoise`, `NormalNoise`)
against the Minecraft classes themselves — the deobfuscated game is on the test classpath, so parity
is asserted without launching the game. Hardware-dependent CUDA tests skip themselves on machines
without an NVIDIA driver.

`ImprovedNoiseGpuTest` is the GPU acceptance test: a million points through the CUDA kernel must
match the Java reference bit for bit.

`ChunkPassProfileTest` prints where a chunk's time goes, per pass. It asserts nothing and skips
without a CUDA device; it is kept because most of the bottlenecks this project chased turned out not
to be the bottleneck, and the only thing that settled any of them was a number.

## Status

| Milestone | State |
|---|---|
| M0 — CPU reference + parity | **done.** The overworld density function is lowered to an instruction program and evaluated bit-identically to vanilla, and every noise it uses round-trips through the export path. |
| M1 — FFM + a kernel | **done.** A million points through the CUDA `ImprovedNoise` match the Java reference bit for bit. |
| M2 — K0/K1/K2 | **done.** `preliminary_surface_level` and every per-chunk marker table (column caches and the 5×5×49 interpolator corner grids) are computed on the device, bit-identically to the CPU reference. |
| M3 — K3/K4 + chunk replay | **done.** The material rules (aquifer and ore veinifier) are ported and reproduce vanilla's block for every one of 294912 blocks across three chunks; `GpuChunkFiller` turns that into a whole chunk's blocks; and `ChunkReplay` writes them into a chunk, with the heightmaps and the fluid-update flags `doFill` also maintains. |
| M4 — batching, pinned buffers | **in progress.** One launch per chunk per program instead of one per marker, and the per-block pass — the trilinear blend and the DAG above it — runs on the device rather than on one core. 16.2 ms per chunk became 7.8 ms, and the mixin takes over generation. Batching across chunks and pinned staging are still open. |

The device path now generates terrain. It is off by default; with it on, the NOISE stage runs on the
GPU and vanilla's own is cancelled. Anything that goes wrong falls through to vanilla, which writes
every block itself.

### Where the work sits

A chunk's density goes through three device passes per program, and the split is deliberate:

- **marker tables** — the `5 × 5 × 49` corner grid of every interpolated marker, one launch carrying
  all of them;
- **the per-block pass** — the corner grids blended down to block resolution and the few dozen
  instructions above them, in one launch that returns a single double per block;
- **material rules** — the aquifer and the ore veinifier, which are branch- and table-driven rather
  than floating-point heavy, and stay on the CPU.

That last choice is the design doc's, and it is measured rather than assumed: moving the blend and
the DAG above the markers to the device was worth about eleven milliseconds a chunk, and moving the
aquifer would mean a per-thread copy of its grid and caches for very little.

Three things happen when a chunk is written, and all three are in vanilla's `doFill`: the block
states, the two worldgen heightmaps, and `markPosForPostprocessing` for the positions the aquifer
flagged as able to flow. The last is easy to leave out and is the difference between water that flows
into a cave and water that sits there. The flags are compared against vanilla's own aquifer, not
smoke-tested: most chunks flag nothing at all, so the test uses three that flag 155, 67 and 145.

### Trying it in a game

The device path is off by default. To run the whole chain — lowering, upload, density, material
rules, block ids — inside a real world without changing what the game generates:

```
./gradlew runClient -Pshadow
```

Add `-Pverbose` to log every chunk rather than every 256th. (`-Pterracuda.shadow` works too, but the
Windows shell splits it at the dot, so the dotless form is the one to use there.) The hook computes each chunk's
blocks on the device path and logs how long that took; it never cancels the vanilla call and never
touches the chunk. A device that cannot be used degrades to a log line.

To let it build the chunks instead:

```
./gradlew runClient -Ptakeover
```

Vanilla's NOISE stage is cancelled and the device's blocks are written into the chunk. On an RTX 3080
a chunk takes about 9 ms at the median, 11 ms at the ninetieth percentile, over 2661 chunks of a real
world with no exceptions. A dimension whose chunks are not the height the generator is configured for,
a legacy-world blend, and anything that fails mid-flight all fall through to vanilla.

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

The implementation plan lives in [`docs/GPU_区块生成规划.md`](docs/GPU_区块生成规划.md).

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
