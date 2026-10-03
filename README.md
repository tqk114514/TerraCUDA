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

## GPU kernels

The CUDA source lives in `src/main/cuda/` and is shipped inside the jar. At runtime the mod picks the
best available image, in this order:

1. a cubin prebuilt for the device's exact compute capability;
2. a prebuilt PTX image, JIT-compiled by the driver;
3. the embedded CUDA source, compiled at runtime with NVRTC.

Rungs 1 and 2 are produced by `nvcc` at build time. On Windows `nvcc` additionally requires the MSVC
host compiler (`cl.exe`); without it, rung 3 still works, so the CUDA Toolkit alone is enough to run
the GPU path — it just costs a one-off JIT at startup instead of none.

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
