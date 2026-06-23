#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <random>
#include "secp.h"
#include "addr.h"

static void scalar_n(scalar* r){
    r->d[0]=0xBFD25E8CD0364141ULL; r->d[1]=0xBAAEDCE6AF48A03BULL;
    r->d[2]=0xFFFFFFFFFFFFFFFEULL; r->d[3]=0xFFFFFFFFFFFFFFFFULL;
}
static int scalar_ge(const scalar* a, const scalar* b){
    for (int i=3;i>=0;i--){ if (a->d[i]>b->d[i]) return 1; if (a->d[i]<b->d[i]) return 0; }
    return 1;
}
static void scalar_sub(scalar* r, const scalar* a, const scalar* b){
    u128 br=0;
    for (int i=0;i<4;i++){ u128 c=(u128)a->d[i]-b->d[i]-br; r->d[i]=(uint64_t)c; br=(c>>64)&1; }
}
static void scalar_reduce_n(scalar* s){
    scalar n; scalar_n(&n);
    for (int k=0;k<4 && scalar_ge(s,&n); k++) scalar_sub(s,s,&n);
}
static void scalar_from_b32(scalar* s, const uint8_t b[32]){
    for (int i=0;i<4;i++){
        uint64_t v=0; const uint8_t* p=b+(3-i)*8;
        for (int k=0;k<8;k++) v=(v<<8)|p[k];
        s->d[i]=v;
    }
}
static void scalar_to_b32(uint8_t b[32], const scalar* s){
    for (int i=0;i<4;i++){ uint64_t v=s->d[3-i]; for (int k=0;k<8;k++) b[i*8+k]=(uint8_t)(v>>(56-8*k)); }
}
static int hex2bin(const char* hex, uint8_t* out, int outlen){
    int n=(int)strlen(hex); if (n!=outlen*2) return -1;
    for (int i=0;i<outlen;i++){
        int hi,lo; char c;
        c=hex[i*2];   hi=(c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:(c>='A'&&c<='F')?c-'A'+10:-1;
        c=hex[i*2+1]; lo=(c>='0'&&c<='9')?c-'0':(c>='a'&&c<='f')?c-'a'+10:(c>='A'&&c<='F')?c-'A'+10:-1;
        if (hi<0||lo<0) return -1;
        out[i]=(uint8_t)((hi<<4)|lo);
    }
    return 0;
}
static void tohex(const uint8_t* b, int len, char* out){
    static const char* H="0123456789abcdef";
    for (int i=0;i<len;i++){ out[i*2]=H[b[i]>>4]; out[i*2+1]=H[b[i]&15]; }
    out[len*2]=0;
}

static int dump_priv(const uint8_t priv_in[32]){
    scalar s; scalar_from_b32(&s, priv_in);
    scalar_reduce_n(&s);
    if ((s.d[0]|s.d[1]|s.d[2]|s.d[3])==0){ fprintf(stderr,"ERROR: zero scalar\n"); return 1; }
    uint8_t priv[32]; scalar_to_b32(priv,&s);

    affine P; scalar_mul_G(&P,&s);
    uint8_t pk[33]; pubkey_compressed(pk,&P);
    uint8_t h160[20]; hash160(pk,33,h160);

    char hpriv[65], hpk[67], hh[41];
    tohex(priv,32,hpriv); tohex(pk,33,hpk); tohex(h160,20,hh);
    std::string a_p2wpkh=p2wpkh_address(h160);
    std::string a_p2pkh =p2pkh_address(h160);
    std::string wif=wif_compressed(priv);

    printf("PRIV %s\n", hpriv);
    printf("PUBKEY %s\n", hpk);
    printf("HASH160 %s\n", hh);
    printf("P2WPKH %s\n", a_p2wpkh.c_str());
    printf("P2PKH %s\n", a_p2pkh.c_str());
    printf("WIF %s\n", wif.c_str());
    return 0;
}

static int walkcheck(long N){

    scalar k0; k0.d[0]=0x123456789abcdef0ULL; k0.d[1]=0x0fedcba987654321ULL;
    k0.d[2]=0xdeadbeefcafebabeULL; k0.d[3]=0x0000000000000042ULL;
    affine base; scalar_mul_G(&base,&k0);
    affine G; G.inf=0; secp_Gx(&G.x); secp_Gy(&G.y);
    affine cur=base;
    long mism=0;
    for (long i=0;i<N;i++){
        scalar ki; scalar_add_u64(&ki,&k0,(uint64_t)i);
        affine ref; scalar_mul_G(&ref,&ki);
        if (!(fe_eq(&cur.x,&ref.x)&&fe_eq(&cur.y,&ref.y))){
            mism++;
            if (mism<=5) fprintf(stderr,"walk mismatch at i=%ld\n",i);
        }
        affine nxt; affine_add(&nxt,&cur,&G);
        cur=nxt;
    }
    printf("walkcheck N=%ld mismatches=%ld %s\n", N, mism, mism==0?"PASS":"FAIL");
    return mism==0?0:1;
}

static int selftest(){

    uint8_t one[32]={0}; one[31]=1;
    scalar s; scalar_from_b32(&s,one);
    affine P; scalar_mul_G(&P,&s);
    fe gx,gy; secp_Gx(&gx); secp_Gy(&gy);
    int ok1=fe_eq(&P.x,&gx)&&fe_eq(&P.y,&gy);
    printf("selftest priv=1==G: %s\n", ok1?"PASS":"FAIL");

    dump_priv(one);

    fe a; a.d[0]=0x0123456789abcdefULL;a.d[1]=0xfedcba9876543210ULL;a.d[2]=7;a.d[3]=0x99;
    fe b; b.d[0]=0xdeadbeefULL;b.d[1]=0x1234;b.d[2]=0;b.d[3]=0;
    fe t,binv,r; fe_mul(&t,&a,&b); fe_inv(&binv,&b); fe_mul(&r,&t,&binv);
    int ok2=fe_eq(&r,&a);
    printf("selftest field inv roundtrip: %s\n", ok2?"PASS":"FAIL");
    return (ok1&&ok2)?0:1;
}

static int gentest(long N, uint64_t seed){
    std::mt19937_64 rng(seed);
    for (long i=0;i<N;i++){
        uint8_t priv[32];
        for (int j=0;j<32;j++) priv[j]=(uint8_t)(rng()&0xff);
        priv[0]&=0x7f;
        scalar s; scalar_from_b32(&s,priv); scalar_reduce_n(&s);
        uint8_t pb[32]; scalar_to_b32(pb,&s);
        affine P; scalar_mul_G(&P,&s);
        uint8_t pk[33]; pubkey_compressed(pk,&P);
        uint8_t h160[20]; hash160(pk,33,h160);
        char hpriv[65],hh[41]; tohex(pb,32,hpriv); tohex(h160,20,hh);
        std::string addr=p2wpkh_address(h160);
        printf("%s %s %s\n", hpriv, hh, addr.c_str());
    }
    return 0;
}

static int cpu_search(const char* prefix){
    std::random_device rd; std::mt19937_64 rng(((uint64_t)rd()<<32)^rd());
    uint8_t priv[32];
    for (int j=0;j<32;j++) priv[j]=(uint8_t)(rng()&0xff);
    scalar s; scalar_from_b32(&s,priv); scalar_reduce_n(&s);
    affine cur; scalar_mul_G(&cur,&s);
    affine G; G.inf=0; secp_Gx(&G.x); secp_Gy(&G.y);
    std::string want=std::string("bc1q")+prefix;
    for (long i=0;;i++){
        uint8_t pk[33]; pubkey_compressed(pk,&cur);
        uint8_t h160[20]; hash160(pk,33,h160);
        std::string a=p2wpkh_address(h160);
        if (a.compare(0,want.size(),want)==0){
            scalar ki; scalar_add_u64(&ki,&s,(uint64_t)i); scalar_reduce_n(&ki);
            uint8_t pb[32]; scalar_to_b32(pb,&ki);
            char hpriv[65]; tohex(pb,32,hpriv);
            printf("FOUND priv=%s addr=%s steps=%ld\n", hpriv, a.c_str(), i);
            return dump_priv(pb);
        }
        affine nxt; affine_add(&nxt,&cur,&G); cur=nxt;
    }
}

int main(int argc, char** argv){
    if (argc>=2 && strcmp(argv[1],"--dump")==0 && argc>=3){
        uint8_t priv[32];
        if (hex2bin(argv[2],priv,32)!=0){ fprintf(stderr,"bad hex (need 64)\n"); return 2; }
        return dump_priv(priv);
    }
    if (argc>=2 && strcmp(argv[1],"--walkcheck")==0){
        long N=argc>=3?atol(argv[2]):1000; return walkcheck(N);
    }
    if (argc>=2 && strcmp(argv[1],"--selftest")==0) return selftest();
    if (argc>=2 && strcmp(argv[1],"--gentest")==0){
        long N=argc>=3?atol(argv[2]):100; uint64_t seed=argc>=4?strtoull(argv[3],0,10):12345;
        return gentest(N,seed);
    }
    if (argc>=2 && strcmp(argv[1],"--search")==0 && argc>=3) return cpu_search(argv[2]);
    fprintf(stderr,"usage: %s --dump <hex> | --walkcheck N | --selftest | --gentest N seed | --search <prefix>\n", argv[0]);
    return 2;
}
