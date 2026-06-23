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
points, `center±i·G`), and register-resident single-block SHA-256/RIPEMD-160.

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
charset `qpzry9x8gf2tvdw0s3jn54khce6mua7l`):

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

## How it works

- One CSPRNG base scalar `k0` is drawn on the host.
- Each thread/window is centered on a scalar; the window covers the `2M+1`
  consecutive scalars around the center (`M = W-1`, `W = 768`).
- The center point is computed once with a scalar multiply; the rest of the
  window is built from a constant-memory table `i·G`.
- **±symmetry:** `center+i·G` and `center−i·G` share the same denominator, so
  one batch inverse (Montgomery trick) serves two points.
- Each point is hashed to a hash160 and its leading bits compared to the target
  derived from the requested bech32 prefix.
- Hits store `(thread, window, signed step)`; the host reconstructs the private
  scalar, recomputes the address, and **rejects any hit that does not
  reproduce** before printing.

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

Run the checks yourself:

```bash
python3 verify.py ./vanity 500                       # CPU oracle vs coincurve
./vanity_gpu --gpufieldtest 2000                     # gate 1
./vanity_gpu --gpuwalkcheck 8192                      # gate 2
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
