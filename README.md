# cuda-vanity-secp256k1

A CUDA GPU **vanity address generator** for Bitcoin's secp256k1 curve. Searches
for a chosen prefix on **bech32 P2WPKH** addresses (`bc1q…`) and derives the
matching **legacy P2PKH** (`1…`) address and WIF. Every result is re-derived and
verified on the host before it is printed.

The elliptic-curve, SHA-256 and RIPEMD-160 code is shared verbatim between the
CPU reference and the GPU kernels (one header, `__host__ __device__`), and the
whole pipeline is validated **bit-for-bit against libsecp256k1** (via
`coincurve`) and the canonical BIP-173 test vectors.

> ~**2.2 Gkey/s** on a single RTX 3090. A "key" is a full
> point → compressed pubkey → SHA-256 → RIPEMD-160 → hash160 → prefix test.

## Performance

| GPU | rate |
|-----|------|
| RTX 3090 (sm_86) | ~2,190 Mkey/s |

Throughput comes from a precomputed `i·G` table + Montgomery batch inversion
(one modular inverse per window), a **±symmetry** walk (each inverse yields two
points, `center±i·G`), register-resident single-block SHA-256/RIPEMD-160, and the
**GLV endomorphism** — each computed point is tested as up to **six** addresses
(`x`, `βx`, `β²x`, each with both y-parities) for the cost of two field multiplies,
multiplying the addresses checked per unit of curve arithmetic.

## Requirements

- NVIDIA GPU and the CUDA toolkit (`nvcc`); developed on CUDA 12.4.
- A C++14 host compiler (`g++`) for the CPU oracle.
- Python 3 with [`coincurve`](https://pypi.org/project/coincurve/) — only needed
  to run the validation scripts.

## Build

```bash
make all ARCH=sm_86      # builds ./vanity (CPU) and ./vanity_gpu (CUDA)
```

Set `ARCH` to your card's compute capability:

| Card family | ARCH |
|-------------|------|
| RTX 20xx (Turing)    | `sm_75` |
| RTX 30xx (Ampere)    | `sm_86` |
| RTX 40xx (Ada)       | `sm_89` |
| RTX 50xx (Blackwell) | `sm_120` (CUDA ≥ 12.8) |

The source is identical across cards — only the `ARCH` flag changes.

## Usage

Search for a bech32 prefix (characters after `bc1q` must be in the bech32
charset `qpzry9x8gf2tvdw0s3jn54khce6mua7l`). Up to **32 characters** after `bc1q`
(160 bits — the full hash160) are supported:

```bash
./vanity_gpu --search bc1qcafe --blocks 2048 --windows 16
```

Example output (every hit is host-verified):

```
HIT VERIFIED
  PRIV    6ee1d513e8b1ba1c52c756e93147e58318513c604f066451c88958efe7cc5018
  HASH160 c631856981625159d237314d2792fad74a7bb20e
  P2WPKH  bc1qcafe...
  P2PKH   1K4x7mkmKG2jeZDmXFX9bXEatEdD8UgB6p
  WIF     KzwFXeVkiKrQDiiAeyhMH8noXBhVKq3vCaQ3H4xpt7dbvofbi7TM
```

Re-derive everything from a private key with the CPU oracle:

```bash
./vanity --dump <64-hex-private-key>
```

Flags: `--blocks N` (thread blocks), `--windows N` (windows per launch),
`--maxsec S` (stop after S seconds with no hit).

### Tuning knobs (compile-time)

The candidate-per-point fan-out is controlled by two macros so you can find the
sweet spot for your card (more candidates = more addresses per point, but more
hashing and register pressure):

```bash
make gpu ARCH=sm_86                              # default: 6 candidates/point (endo + parity)
make gpu ARCH=sm_86 NVCCFLAGS="-O3 -arch=sm_86 -std=c++14 --expt-relaxed-constexpr -DUSE_ENDO=0"   # 2/point (parity only)
make gpu ARCH=sm_86 NVCCFLAGS="-O3 -arch=sm_86 -std=c++14 --expt-relaxed-constexpr -DUSE_ENDO=0 -DUSE_PARITY=0"  # 1/point (original)
```

| `USE_ENDO` | `USE_PARITY` | candidates / point | notes |
|-----------|--------------|--------------------|-------|
| 1 | 1 | 6 | default (GLV β,β² × both parities) |
| 1 | 0 | 3 | GLV images, single parity |
| 0 | 1 | 2 | y-symmetry only, no GLV muls |
| 0 | 0 | 1 | original behaviour |

`W` (window/group size) is also a compile knob (`-DWSIZE=N`, default 768). If the
6-candidate kernel is register/occupancy-bound, try a smaller `W` together with
the full fan-out, or dial the fan-out down. Sweep `--blocks` / `--windows` at
runtime as before.

## How it works

- One CSPRNG base scalar `k0` is drawn on the host.
- Each thread/window is centered on a scalar; the window covers the `2M+1`
  consecutive scalars around the center (`M = W-1`, `W = 768`).
- The center point is computed once with a scalar multiply; the rest of the
  window is built from a constant-memory table `i·G`.
- **±symmetry:** `center+i·G` and `center−i·G` share the same denominator, so
  one batch inverse (Montgomery trick) serves two points.
- **GLV endomorphism:** for each computed affine point `(x, y)`, the curve map
  `[λ](x,y) = (βx, y)` gives two more valid public keys (`βx`, `β²x`) for the
  cost of two field multiplies, and negating `y` (flip the compressed-pubkey
  parity byte, `s → n−s`) doubles that again — up to **6 candidate addresses per
  point**. Each candidate corresponds to a recoverable private key
  `(−1)^neg · λ^e · s (mod n)`.
- Each candidate is hashed to a hash160 and its leading bits compared to the
  target derived from the requested bech32 prefix (up to 160 bits / 32 chars,
  matched across three 64-bit words; short prefixes short-circuit on the first).
- Hits store `(thread, window, signed step, variant)`; the host reconstructs the
  base scalar, applies the variant transform to get the real private key,
  recomputes the address, and **rejects any hit that does not reproduce** before
  printing.

## Correctness

The implementation is validated by five gates (see `BENCH.md`):

1. **Field equivalence** — `fe_mul/fe_sqr/fe_inv/fe_add/fe_sub/reduce` are
   bit-identical on GPU and CPU over thousands of inputs.
2. **GPU walk** — the batch-inversion window equals an independent
   `scalar_mul_G(center±i)` for thousands of consecutive scalars.
3. **Hit reproduction** — every emitted hit reproduces through the CPU oracle
   and `coincurve` (address + WIF match exactly).
4. **Cross-check** — GPU-derived addresses match `coincurve` 100%.
5. **bech32** — charset and checksum match an independent encoder and the
   canonical BIP-173 vector (`priv=1 → bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4`).

6. **GLV endomorphism** — for thousands of scalars and all six variants, the
   address built from `(β^e·x, ±y)` equals the address derived from the recovered
   key `(−1)^neg·λ^e·s`. Checked on the host (`--endotest`, no GPU needed) and on
   the device (`--gpuendocheck`).

Run the checks yourself:

```bash
./vanity --endotest 5000                             # gate 6, host (GLV math + key recovery)
python3 verify.py ./vanity 500                       # CPU oracle vs coincurve
./vanity_gpu --gpufieldtest 2000                     # gate 1
./vanity_gpu --gpuwalkcheck 8192                      # gate 2
./vanity_gpu --gpuendocheck 8192                      # gate 6, device (GLV path on GPU)
./vanity_gpu --gpuderive 2000 | python3 gpucheck.py --gen   # gate 4
```

## Security notes — read before generating keys for real funds

- **RNG.** The base scalar is seeded from `std::random_device` and expanded with
  `mt19937_64`, which is **not** a guaranteed cryptographic RNG on every
  platform. Treat this as an educational/research tool. For keys that will hold
  value, replace the RNG with a vetted CSPRNG and/or only import a key after
  independent verification.
- **Trust.** Generated private keys are printed to stdout. Run on a trusted,
  offline machine; never paste a private key into an untrusted service.
- **Verify before funding.** Always confirm a found address/WIF in an
  independent wallet before sending funds. `./vanity --dump <priv>` and
  `coincurve` give two independent derivations.
- **Scope.** bech32 **v0** P2WPKH only — not Taproot / bech32m.

## License

MIT — see [LICENSE](LICENSE).
