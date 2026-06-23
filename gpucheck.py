import sys, hashlib
from coincurve import PublicKey

from verify import ripemd160, p2wpkh, p2pkh, wif

def hash160(b):
    return ripemd160(hashlib.sha256(b).digest())

def ref(priv_hex):
    priv = bytes.fromhex(priv_hex)
    pub = PublicKey.from_valid_secret(priv).format(compressed=True)
    h = hash160(pub)
    return pub.hex(), h.hex(), p2wpkh(h), p2pkh(h), wif(priv)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--gen"
    total = 0; ok = 0; fails = []
    if mode == "--gen":
        for line in sys.stdin:
            p = line.split()
            if len(p) != 3: continue
            priv_hex, h_hex, addr = p
            _, rh, ra, _, _ = ref(priv_hex)
            total += 1
            if rh == h_hex and ra == addr: ok += 1
            else: fails.append((priv_hex, f"h160 {rh}/{h_hex} addr {ra}/{addr}"))
    elif mode == "--dumpcheck":

        cur = {}
        def flush(cur):
            nonlocal total, ok
            if "PRIV" not in cur: return
            total += 1
            pub, rh, rw, rp, rwif = ref(cur["PRIV"])
            good = (cur.get("HASH160")==rh and cur.get("P2WPKH")==rw and
                    cur.get("P2PKH")==rp and cur.get("WIF")==rwif)
            if good: ok += 1
            else: fails.append((cur["PRIV"], f"want {rh}/{rw}/{rp}/{rwif} got {cur}"))
        for line in sys.stdin:
            t = line.strip().split()
            if len(t) == 2 and t[0] in ("PRIV","HASH160","P2WPKH","P2PKH","WIF"):
                if t[0] == "PRIV" and cur: flush(cur); cur = {}
                cur[t[0]] = t[1]
        flush(cur)
    print(f"gpucheck[{mode}]: {ok}/{total} match")
    for pv, msg in fails[:10]:
        print("  FAIL", pv, msg)
    if total > 0 and ok == total:
        print("ALL PASS"); sys.exit(0)
    print("FAILURES PRESENT"); sys.exit(1)

if __name__ == "__main__":
    main()
