# BENCH.md — secp256k1 CUDA vanity generator

## Hardware (rented Vast.ai instance)

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 3090** (Ampere, compute capability **8.6** → `-arch=sm_86`) |
| VRAM | 24 GB |
| Driver | 570.211.01 |
| CUDA toolkit | 12.4 (V12.4.131), `nvcc` |
| Host compiler | g++ 11.4.0 (Ubuntu 22.04) |
| Vast.ai instance id | **42264745**, label `vanity-gpu-builder` |
| Rate | ~$0.156/hr |

## Throughput

Measured with `--search bc1qqqqqqqqqqqq` (an unreachable 55-bit prefix, so the
kernel never early-exits and we measure pure key-rate). Each "key" is one full
pipeline: point on curve → compressed pubkey → SHA-256 → RIPEMD-160 → hash160
prefix test.

**Sustained: ~2.19 Gkey/s** on the RTX 3090 (default `W=768`, `blocks=2048`,
`windows=16`).

| config | rate |
|--------|------|
| blocks=2048, windows=16 (default) | **2189 Mkey/s** |
| blocks=2048, windows=32 | 2184 Mkey/s |
| blocks=3072, windows=16 | 2157 Mkey/s |
| blocks=1024, windows=32 | 2129 Mkey/s |

### Optimization history (all kept every validation gate green)

| step | rate | factor |
|------|------|--------|
| initial (generic hashes, W=256) | 320 Mkey/s | 1.0× |
| register-resident single-block SHA-256 + RIPEMD-160 | 1520 Mkey/s | 4.7× |
| retune window size to W=768 | 1860 Mkey/s | 5.8× |
| ±symmetry walk (one inverse → two points) | **2189 Mkey/s** | **6.8×** |

1. **Hashing was 78% of per-point cost and the bottleneck.** The generic
   hash functions used byte-indexed `uint8_t buf[128]` scratch arrays that
   spilled to *local* memory; every byte access went off-register. Replacing
   them with specialized, fully-unrolled single-block `sha256_33` /
   `ripemd160_32` (padding baked into the message words, schedule kept in a
   16-word register window) kept everything in registers → **4.7×**. Confirmed
   bit-identical to the generic path and to coincurve (gate 4).
2. **Window retune.** Once hashing was cheap, the batch-inversion amortization
   dominated, so a larger window helped: W swept 64→768. `W=768` is the
   practical max — the `i·G` affine table lives in 64 KB constant memory at
   72 B/entry (768×72 = 55 KB; 1024 would overflow). → **5.8×** total.
3. **±symmetry.** With hashing fast, a `NOHASH` measurement showed field
   arithmetic was 66% of per-point cost. The window was restructured around a
   *center* point `Pc`, computing the pair `Pc+i·G` and `Pc−i·G` together: both
   share the same `dx_i = (Gt[i].x − Pc.x)`, so one batch inverse and one shared
   reciprocal serve two points (field muls/point 6 → 4.5). Hits now store a
   *signed* step from the center; the host reconstructs `center + step`. → **6.8×**.

Config tuning (register caps, smaller W, TPB) was confirmed *not* to help — the
kernel is compute-bound on the field arithmetic + two hashes.

## Algorithm

Verbatim reuse of the CPU field/curve/hash arithmetic (`secp.h`, marked
`__host__ __device__`). The GPU adds no new math — only parallel scheduling.

- Host draws one CSPRNG 256-bit base scalar `k0`.
- Each window is centered on a scalar; with `M = W−1` and `span = 2M+1`, thread
  `t`, window `w` is centered at `k0 + (t + w·nThreads)·span + M` and covers the
  `span` consecutive scalars `[center−M, center+M]` — a contiguous,
  non-overlapping key space.
- Each thread computes its window-0 center point `Pc` with `scalar_mul_G`
  (Jacobian double-and-add), then builds the window from a precomputed affine
  table `c_Gt[i] = i·G` (constant memory).
- **±symmetry:** for each `i = 1..M`, the pair `Pc+i·G` and `Pc−i·G` share the
  same denominator `dx_i = (Gt[i].x − Pc.x)` (since `−i·G` has the same x). So
  all `M` denominators are batch-inverted with **one** `fe_inv` (Montgomery
  trick), and each reciprocal yields two points.
- Each point → compressed pubkey → SHA-256 → RIPEMD-160 → hash160; leading
  bits are compared to the target derived from the bech32 prefix.
- Center advances by `(nThreads·span)·G` (one affine add) between windows.
- Hits store `(tid, widx, signed step)`; the host re-derives
  `center + step`, recomputes the point + hash160, and **verifies before
  printing**. Non-reproducing hits are flagged `REJECTED` (none occurred).

`W = 768` (group size, default), `TPB = 256`. Throughput scales with
`windows/launch` because it amortizes the per-thread startup `scalar_mul_G`.
`W` is a compile-time knob (`-DWSIZE=N`).

## Validation gates (all passed on the RTX 3090)

1. **Field equivalence** — `fe_mul/fe_sqr/fe_inv/fe_add/fe_sub/fe_reduce512`
   bit-identical GPU vs host on 3000 fixed+random inputs (incl. `p−1`, `p`, `1`):
   `gpufieldtest n=3000 mismatches=0 PASS`.
2. **GPU walk** — device ±symmetry window (center, +i, −i) == device
   `scalar_mul_G(center±i)` for 10000 consecutive scalars:
   `gpuwalkcheck N=10000 mismatches=0 PASS`.
3. **Hit reproduction** — 3145 search hits each re-derived through the CPU
   oracle (`./vanity --dump`) and coincurve: `ALL PASS`, 0 rejected,
   0 prefix mismatches.
4. **Cross-check** — 3000 GPU-derived `(priv → hash160 → P2WPKH)` vs coincurve:
   `3000/3000 match — ALL PASS`.
5. **bech32** — charset `qpzry9x8gf2tvdw0s3jn54khce6mua7l`, v0 (checksum const
   `^1`, not bech32m); every produced address matches coincurve + an independent
   Python encoder. Anchored to the canonical BIP-173 vector: priv=1 →
   `bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4`.

## Reproduce

```bash
make all ARCH=sm_86            # builds ./vanity (CPU) and ./vanity_gpu (CUDA)
python3 verify.py ./vanity 500 # CPU oracle vs coincurve -> ALL PASS
./vanity_gpu --gpufieldtest 2000
./vanity_gpu --gpuwalkcheck 8192
./vanity_gpu --gpuderive 2000 31337 | python3 gpucheck.py --gen
./vanity_gpu --search bc1qcccc --blocks 1024 --windows 8 --maxsec 120
```
