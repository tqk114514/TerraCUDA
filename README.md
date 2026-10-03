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

## Status

| Milestone | State |
|---|---|
| M0 — CPU reference + parity | **done.** The overworld density function is lowered to an instruction program and evaluated bit-identically to vanilla, and every noise it uses round-trips through the export path. |
| M1 — FFM + a kernel | **done.** A million points through the CUDA `ImprovedNoise` match the Java reference bit for bit. |
| M2 — K0/K1/K2 | **done.** `preliminary_surface_level` and every per-chunk marker table (column caches and the 5×5×49 interpolator corner grids) are computed on the device, bit-identically to the CPU reference. |
| M3 — K3/K4 + chunk replay | not started. Material rules, aquifer, ore veins, and writing the chunk back. |
| M4 — batching, pinned buffers | not started. |

The GPU path does not yet produce chunks: it computes the density tables, and vanilla's own
interpolation and block writing still do the rest. That is deliberate — a half-filled chunk would be
worse than not running at all.

### Trying it in a game

The device path is off by default. To exercise it inside a running world without changing what the
game generates:

```
-Dterracuda.gpu=true -Dterracuda.shadow=true
```

It computes each chunk's marker tables on the device and logs how long that took. A device that
cannot be used degrades to a log line; nothing throws into the generation path.

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
