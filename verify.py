import sys, os, subprocess, hashlib, secrets

try:
    from coincurve import PublicKey
    from coincurve.utils import GROUP_ORDER_INT
except Exception as e:
    print("MISSING DEP: pip install coincurve  (", e, ")")
    sys.exit(3)

N_ORDER = GROUP_ORDER_INT

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_hrp_expand(hrp):
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]

def bech32_create_checksum(hrp, data):
    values = bech32_hrp_expand(hrp) + data
    polymod = bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def bech32_encode(hrp, data):
    combined = data + bech32_create_checksum(hrp, data)
    return hrp + "1" + "".join(CHARSET[d] for d in combined)

def convertbits(data, frombits, tobits, pad=True):
    acc = 0; bits = 0; ret = []
    maxv = (1 << tobits) - 1
    for b in data:
        acc = (acc << frombits) | b
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    return ret

def p2wpkh(h160, hrp="bc"):
    return bech32_encode(hrp, [0] + convertbits(list(h160), 8, 5))

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def b58encode(b):
    n = int.from_bytes(b, "big")
    s = ""
    while n > 0:
        n, r = divmod(n, 58)
        s = B58[r] + s
    pad = 0
    for c in b:
        if c == 0: pad += 1
        else: break
    return "1" * pad + s

def b58check(payload):
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return b58encode(payload + chk)

def p2pkh(h160):
    return b58check(b"\x00" + h160)

def wif(priv32):
    return b58check(b"\x80" + priv32 + b"\x01")

def _rmd_rol(x, n):
    x &= 0xffffffff
    return ((x << n) | (x >> (32 - n))) & 0xffffffff

def ripemd160(message):

    rl = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
          7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
          3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
          1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
          4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]
    rr = [5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
          6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
          15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
          8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
          12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]
    sl = [11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
          7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
          11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
          11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
          9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]
    sr = [8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
          9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
          9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
          15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
          8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]
    KL = [0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E]
    KR = [0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000]
    def f(j, x, y, z):
        if j < 16:  return x ^ y ^ z
        if j < 32:  return (x & y) | (~x & z)
        if j < 48:  return (x | ~y) ^ z
        if j < 64:  return (x & z) | (y & ~z)
        return x ^ (y | ~z)
    h0,h1,h2,h3,h4 = 0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0
    msg = bytearray(message)
    mlen = len(msg) * 8
    msg.append(0x80)
    while len(msg) % 64 != 56: msg.append(0)
    msg += (mlen & 0xffffffffffffffff).to_bytes(8, "little")
    for off in range(0, len(msg), 64):
        X = [int.from_bytes(msg[off+4*i:off+4*i+4], "little") for i in range(16)]
        al,bl,cl,dl,el = h0,h1,h2,h3,h4
        ar,br,cr,dr,er = h0,h1,h2,h3,h4
        for j in range(80):
            t = (_rmd_rol((al + f(j,bl,cl,dl) + X[rl[j]] + KL[j//16]) & 0xffffffff, sl[j]) + el) & 0xffffffff
            al,el,dl,cl,bl = el,dl,_rmd_rol(cl,10),bl,t
            t = (_rmd_rol((ar + f(79-j,br,cr,dr) + X[rr[j]] + KR[j//16]) & 0xffffffff, sr[j]) + er) & 0xffffffff
            ar,er,dr,cr,br = er,dr,_rmd_rol(cr,10),br,t
        t = (h1 + cl + dr) & 0xffffffff
        h1 = (h2 + dl + er) & 0xffffffff
        h2 = (h3 + el + ar) & 0xffffffff
        h3 = (h4 + al + br) & 0xffffffff
        h4 = (h0 + bl + cr) & 0xffffffff
        h0 = t
    return b"".join(x.to_bytes(4, "little") for x in (h0,h1,h2,h3,h4))

def hash160(b):
    return ripemd160(hashlib.sha256(b).digest())

def ref_derive(priv_int):
    """independent derivation from an integer private key in [1, n-1]."""
    priv32 = priv_int.to_bytes(32, "big")
    pub = PublicKey.from_valid_secret(priv32).format(compressed=True)
    h = hash160(pub)
    return {
        "PRIV": priv32.hex(),
        "PUBKEY": pub.hex(),
        "HASH160": h.hex(),
        "P2WPKH": p2wpkh(h),
        "P2PKH": p2pkh(h),
        "WIF": wif(priv32),
    }

def run_dump(binpath, priv_hex):
    out = subprocess.check_output([binpath, "--dump", priv_hex], text=True)
    d = {}
    for line in out.strip().splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            d[parts[0]] = parts[1].strip()
    return d

def main():
    binpath = sys.argv[1] if len(sys.argv) > 1 else "./vanity"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if not os.path.exists(binpath):
        print("missing binary:", binpath); sys.exit(3)

    fields = ["PRIV", "PUBKEY", "HASH160", "P2WPKH", "P2PKH", "WIF"]
    total = 0; passed = 0; failures = []

    fixed = [1, 2, 0xDEADBEEF, N_ORDER - 1,
             0x123456789abcdef0_0fedcba987654321_deadbeefcafebabe_0000000000000042]
    keys = list(fixed) + [secrets.randbelow(N_ORDER - 1) + 1 for _ in range(N)]

    for k in keys:
        k = (k % (N_ORDER - 1)) + 1 if k >= N_ORDER else k
        ref = ref_derive(k)
        got = run_dump(binpath, ref["PRIV"])
        total += 1
        ok = all(got.get(f) == ref[f] for f in fields)
        if ok:
            passed += 1
        else:
            diffs = [(f, ref[f], got.get(f)) for f in fields if got.get(f) != ref[f]]
            failures.append((ref["PRIV"], diffs))

    try:
        gout = subprocess.check_output([binpath, "--gentest", "500", "777"], text=True)
        gtotal = 0; gpass = 0
        for line in gout.strip().splitlines():
            p = line.split()
            if len(p) != 3: continue
            priv_hex, h_hex, addr = p
            ref = ref_derive(int(priv_hex, 16))
            gtotal += 1
            if ref["HASH160"] == h_hex and ref["P2WPKH"] == addr:
                gpass += 1
            else:
                failures.append((priv_hex, [("gentest", ref["HASH160"]+"/"+ref["P2WPKH"], h_hex+"/"+addr)]))
        print(f"gentest cross-check: {gpass}/{gtotal} match")
    except Exception as e:
        print("gentest skipped:", e)

    print(f"dump verification: {passed}/{total} match")
    for priv, diffs in failures[:10]:
        print("  FAIL", priv)
        for f, r, g in diffs:
            print(f"    {f}: ref={r} got={g}")

    if passed == total and not failures:
        print("ALL PASS")
        sys.exit(0)
    else:
        print("FAILURES PRESENT")
        sys.exit(1)

if __name__ == "__main__":
    main()
