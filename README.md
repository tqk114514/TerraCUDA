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
| M4 — batching, pinned buffers | **partly done, partly declined, partly open.** See below. |

The device path now generates terrain. It is off by default; with it on, the NOISE stage runs on the
GPU and vanilla's own is cancelled. Anything that goes wrong falls through to vanilla, which writes
every block itself.

### M4: what is done, what was declined, what is open

The plan defines M4 as "batching + pinned double-buffering + halo merging", with two acceptance
criteria: amortising a batch of at least eight chunks, and a P95 latency report. "In progress" hid
the fact that three of those four were never started, so they are listed here.

**Done.** One launch per chunk per program instead of one per marker set; the per-block pass moved
from the host onto the device; the dispatcher no longer blocks; the host staging buffers are reused
instead of re-faulted. Together: 16.2 ms per chunk to 6.7 ms, and 40 chunks a second to 64.

**Batching — measured, then declined.** Folding eight chunks into one marker-table launch is worth
0.7 ms a chunk (1.55 → 0.97) at the cost of a scratch that grows from 25 MB to 200 MB, and the
device is not the constraint anyway: its share of the work allows 149 chunks a second against 64
delivered. It would raise a ceiling nobody is touching. The numbers are in `ChunkPassProfileTest`.

**Pinned double-buffering — not started.** The device-to-host traffic is three 786 KB copies a chunk,
about 0.24 ms of the 6.7, measured at 9.7 GB/s through pageable memory. Pinned buffers might double
that, so the whole prize is under 2%. It is the cheapest of the three to try and the smallest.

**Halo merging — not started, and not independently valuable.** A chunk's corner grid is 5×5 cell
columns, and the far row and column belong to the neighbouring chunks: nine of the twenty-five are
computed twice. Deduplicating them would save roughly a third of the marker-table work — but only if
several chunks are in flight at once, which is the batching above, with the same scratch. It is a
refinement of a thing already declined.

**Acceptance criteria.** "At least eight chunks amortised" is not met, and that is the recorded
decision above rather than an omission. "P95 latency report" is not met: the timing hook accumulates
sums, not samples, so it can produce means and not percentiles. Producing one means keeping the
samples, which is a small change to the timing hook and a re-run.

### Where the time goes, and where it does not

On an RTX 3080, a chunk is about 6.7 ms on the device path: 2.1 ms of marker tables, 0.9 ms of
per-block passes, 2.2 ms of material rules, and the rest host-side. Measured over a real world, the
path sustains about **64 chunks a second** — against a GPU whose share of that work would allow
**149**. The device is not the bottleneck; the rest of the chunk pipeline is, and it shares the same
four dispatcher threads.

That is why the design doc's batching is not built. Its premise is that a chunk's marker tables are
latency-bound — 6450 points against a 450-step chain is about 5% occupancy — so folding several
chunks into one launch should be nearly free. Measuring it says otherwise:

| chunks per launch | total | per chunk |
|---|---|---|
| 1 | 1.55 ms | 1.55 ms |
| 4 | 4.00 ms | 1.00 ms |
| 8 | 7.77 ms | 0.97 ms |
| 16 | 13.71 ms | 0.86 ms |

Sixteen times the threads buys 1.8x, so the kernel is not simply waiting on occupancy. The scratch
grows linearly with the batch, so eight chunks costs 200 MB to save 0.7 ms a chunk — and against a
GPU with 2x headroom, it would not show up at all. `ChunkPassProfileTest` prints these numbers and
anyone can re-run them.

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
a chunk takes about 6.7 ms and the path sustains roughly 64 chunks a second — over 2886 chunks of a
real world, with no exceptions and no fallbacks to vanilla. A dimension whose chunks are not the
height the generator is configured for, a legacy-world blend, and anything that fails mid-flight all
fall through to vanilla.

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
