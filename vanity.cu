#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include "secp.h"
#include "addr.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

#ifndef WSIZE
#define WSIZE 768

#endif
#define W WSIZE
#ifndef TPB
#define TPB 256
#endif

__constant__ affine c_Gt[W];
__constant__ affine c_Sstride;
__constant__ fe       c_beta;               // GLV endomorphism constant (beta in F_p)
#define TARGW 3                              // up to 160-bit (32-char) prefix targets
__constant__ uint64_t c_target[TARGW];
__constant__ uint64_t c_mask[TARGW];

// ---- variant knobs (compile-time, for benchmarking / bisecting on real HW) ----
//   USE_ENDO=1   -> also test (beta*x, y) and (beta^2*x, y)   (GLV, +2 muls, x2 hashes)
//   USE_PARITY=1 -> also test the y-negation of each x         (+0 muls, x2 hashes)
//   default 6 candidates/point; set either to 0 to shrink if register/hash-bound.
#ifndef USE_ENDO
#define USE_ENDO 1
#endif
#ifndef USE_PARITY
#define USE_PARITY 1
#endif
#if USE_ENDO
#define NENDO 3
#else
#define NENDO 1
#endif
#if USE_PARITY
#define NSIGN 2
#else
#define NSIGN 1
#endif

struct Hit { uint32_t tid, widx; int32_t step; uint8_t variant; uint8_t h160[20]; };
#define MAXHITS 4096
__device__ Hit  d_hits[MAXHITS];
__device__ unsigned int d_hitcount;

__device__ __forceinline__ int h160_match(const uint8_t h[20]){
    uint64_t w0=0,w1=0,w2=0;
    #pragma unroll
    for (int i=0;i<8;i++)  w0=(w0<<8)|h[i];
    #pragma unroll
    for (int i=8;i<16;i++) w1=(w1<<8)|h[i];
    #pragma unroll
    for (int i=16;i<20;i++)w2=(w2<<8)|h[i];
    w2<<=32;
    if ((w0 & c_mask[0]) != c_target[0]) return 0;   // short prefixes: mask1=mask2=0 -> trivially pass
    if ((w1 & c_mask[1]) != c_target[1]) return 0;
    if ((w2 & c_mask[2]) != c_target[2]) return 0;
    return 1;
}

struct FieldIn  { fe a, b; uint64_t t[8]; };
struct FieldOut { fe mul, sqr, inv, add, sub, red; };

__global__ void k_field_test(const FieldIn* in, FieldOut* out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    fe a=in[i].a, b=in[i].b;
    fe_mul(&out[i].mul,&a,&b);
    fe_sqr(&out[i].sqr,&a);
    if (!fe_is_zero(&a)) fe_inv(&out[i].inv,&a); else fe_set_zero(&out[i].inv);
    fe_add(&out[i].add,&a,&b);
    fe_sub(&out[i].sub,&a,&b);
    uint64_t t[8];
    #pragma unroll
    for (int k=0;k<8;k++) t[k]=in[i].t[k];
    fe_reduce512(&out[i].red,t);
}

__global__ void k_walkcheck(scalar k0, long N, unsigned long long* mism){
    if (blockIdx.x*blockDim.x+threadIdx.x != 0) return;
    *mism=0;
    const uint32_t M=W-1;
    const uint64_t span=2ULL*M+1;
    long done=0; long wc=0;
    while (done<N){
        scalar cs; scalar_add_u64(&cs,&k0,(uint64_t)wc*span + M);
        affine Pc; scalar_mul_G(&Pc,&cs);

        fe pp[W];
        fe acc; fe_set_one(&acc);
        for (uint32_t i=1;i<=M;i++){ fe d; fe_sub(&d,&c_Gt[i].x,&Pc.x); fe_mul(&acc,&acc,&d); pp[i]=acc; }
        fe invall; fe_inv(&invall,&acc);

        { affine ref; scalar_mul_G(&ref,&cs);
          if (!(fe_eq(&Pc.x,&ref.x)&&fe_eq(&Pc.y,&ref.y))) atomicAdd(mism,1ULL); done++; }
        for (int i=(int)M;i>=1 && done<N;i--){
            fe d; fe_sub(&d,&c_Gt[i].x,&Pc.x);
            fe prev; if (i>=2) prev=pp[i-1]; else fe_set_one(&prev);
            fe invd; fe_mul(&invd,&prev,&invall);
            fe_mul(&invall,&invall,&d);

            { fe lam,num,x3,t0,y3;
              fe_sub(&num,&c_Gt[i].y,&Pc.y); fe_mul(&lam,&num,&invd);
              fe_sqr(&x3,&lam); fe_sub(&x3,&x3,&Pc.x); fe_sub(&x3,&x3,&c_Gt[i].x);
              fe_sub(&t0,&Pc.x,&x3); fe_mul(&t0,&lam,&t0); fe_sub(&y3,&t0,&Pc.y);
              scalar ki; scalar_add_u64(&ki,&cs,(uint64_t)i); affine ref; scalar_mul_G(&ref,&ki);
              if (!(fe_eq(&x3,&ref.x)&&fe_eq(&y3,&ref.y))) atomicAdd(mism,1ULL); done++; }

            { fe lam,num,ny,x3,t0,y3;
              fe_negate(&ny,&c_Gt[i].y);
              fe_sub(&num,&ny,&Pc.y); fe_mul(&lam,&num,&invd);
              fe_sqr(&x3,&lam); fe_sub(&x3,&x3,&Pc.x); fe_sub(&x3,&x3,&c_Gt[i].x);
              fe_sub(&t0,&Pc.x,&x3); fe_mul(&t0,&lam,&t0); fe_sub(&y3,&t0,&Pc.y);
              scalar ki; scalar_sub_u64(&ki,&cs,(uint64_t)i); affine ref; scalar_mul_G(&ref,&ki);
              if (!(fe_eq(&x3,&ref.x)&&fe_eq(&y3,&ref.y))) atomicAdd(mism,1ULL); done++; }
        }
        wc++;
    }
}

__global__ void k_derive(const scalar* ks, uint8_t* h160out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    affine P; scalar s=ks[i]; scalar_mul_G(&P,&s);
    uint8_t h[20]; pubkey_hash160(h,&P);
    for (int k=0;k<20;k++) h160out[i*20+k]=h[k];
}

// Self-test for the GLV path: for N consecutive scalars and all 6 variants, the
// candidate hash built from (beta^e * x, sign*y) must equal the hash of the point
// derived from the recovered key (-1)^neg * lambda^e * s. Any mismatch -> count.
__global__ void k_endocheck(scalar k0, long N, unsigned long long* mism){
    if (blockIdx.x*blockDim.x+threadIdx.x != 0) return;
    *mism=0;
    scalar lambda; secp_lambda(&lambda);
    for (long it=0; it<N; it++){
        scalar s; scalar_add_u64(&s,&k0,(uint64_t)it);
        affine P; scalar_mul_G(&P,&s);
        uint8_t par=(uint8_t)fe_is_odd(&P.y);
        fe xs[3]; xs[0]=P.x; fe_mul(&xs[1],&P.x,&c_beta); fe_mul(&xs[2],&xs[1],&c_beta);
        for (int e=0;e<3;e++) for (int neg=0;neg<2;neg++){
            uint8_t pk[33]; pk[0]=0x02 | (par ^ (uint8_t)neg); fe_get_b32(pk+1,&xs[e]);
            uint32_t sh[8]; sha256_33(pk,sh); uint8_t hc[20]; ripemd160_32(sh,hc);
            scalar key=s; for(int q=0;q<e;q++) scn_mul(&key,&key,&lambda); if(neg) scn_neg(&key,&key);
            affine R; scalar_mul_G(&R,&key); uint8_t hr[20]; pubkey_hash160(hr,&R);
            int ok=1; for(int k=0;k<20;k++) if(hc[k]!=hr[k]) ok=0;
            if(!ok) atomicAdd(mism,1ULL);
        }
    }
}

__device__ __forceinline__ void store_hit(uint32_t t, uint32_t w, int32_t step,
                                          uint8_t variant, const uint8_t h[20]){
    unsigned int slot=atomicAdd(&d_hitcount,1u);
    if (slot<MAXHITS){
        d_hits[slot].tid=t; d_hits[slot].widx=w; d_hits[slot].step=step; d_hits[slot].variant=variant;
        #pragma unroll
        for (int k=0;k<20;k++) d_hits[slot].h160[k]=h[k];
    }
}

// Test up to NENDO*NSIGN candidate addresses derived from a single point Q:
//   endomorphism images   x, beta*x, beta^2*x         (private keys s, lam*s, lam^2*s)
//   y-negation per image  flips the parity byte         (private key -> n - key)
// On a hit, the (endo,sign) index is packed into the variant byte so the host can
// recover the exact private key and re-verify before printing.
__device__ __forceinline__ void check_point(const affine* Q, uint32_t t, uint32_t w, int32_t step){
    fe xs[NENDO];
    xs[0]=Q->x;
#if NENDO>1
    fe_mul(&xs[1],&Q->x,&c_beta);
#endif
#if NENDO>2
    fe_mul(&xs[2],&xs[1],&c_beta);
#endif
#ifdef NOHASH
    // field-only benchmark path: do the endomorphism muls but skip hashing
    if ((xs[NENDO-1].d[3] & c_mask[0]) == c_target[0]){
        uint8_t z[20];
        #pragma unroll
        for (int k=0;k<20;k++) z[k]=0;
        store_hit(t,w,step,0,z);
    }
#else
    uint8_t par=(uint8_t)fe_is_odd(&Q->y);
    // outer loop NOT force-unrolled: inlining the hash NENDO times spikes register
    // pressure and hurts occupancy (pp[W] already pins a lot of local memory).
    // The inner sign loop IS unrolled — both parities reuse the same serialized pk[1..32].
    for (int e=0;e<NENDO;e++){
        uint8_t pk[33];
        fe_get_b32(pk+1,&xs[e]);
        #pragma unroll
        for (int neg=0;neg<NSIGN;neg++){
            pk[0]=0x02 | (par ^ (uint8_t)neg);
            uint32_t sh[8]; sha256_33(pk,sh);
            uint8_t h[20];  ripemd160_32(sh,h);
            if (h160_match(h)) store_hit(t,w,step,(uint8_t)(e*2+neg),h);
        }
    }
#endif
}

__global__ void __launch_bounds__(TPB) k_search(scalar k0, uint32_t nThreads, uint32_t windows){
    uint32_t t=blockIdx.x*blockDim.x+threadIdx.x;
    if (t>=nThreads) return;
    const uint32_t M=W-1;
    const uint64_t span=2ULL*M+1;

    scalar cs; scalar_add_u64(&cs,&k0,(uint64_t)t*span + M);
    affine Pc; scalar_mul_G(&Pc,&cs);
    for (uint32_t w=0; w<windows; w++){

        fe pp[W];
        fe acc; fe_set_one(&acc);
        for (uint32_t i=1;i<=M;i++){ fe d; fe_sub(&d,&c_Gt[i].x,&Pc.x); fe_mul(&acc,&acc,&d); pp[i]=acc; }
        fe invall; fe_inv(&invall,&acc);

        check_point(&Pc,t,w,0);
        for (int i=(int)M;i>=1;i--){
            fe d; fe_sub(&d,&c_Gt[i].x,&Pc.x);
            fe prev; if (i>=2) prev=pp[i-1]; else fe_set_one(&prev);
            fe invd; fe_mul(&invd,&prev,&invall);
            fe_mul(&invall,&invall,&d);

            { fe lam,num,x3,t0,y3;
              fe_sub(&num,&c_Gt[i].y,&Pc.y); fe_mul(&lam,&num,&invd);
              fe_sqr(&x3,&lam); fe_sub(&x3,&x3,&Pc.x); fe_sub(&x3,&x3,&c_Gt[i].x);
              fe_sub(&t0,&Pc.x,&x3); fe_mul(&t0,&lam,&t0); fe_sub(&y3,&t0,&Pc.y);
              affine Q; Q.inf=0; Q.x=x3; Q.y=y3; check_point(&Q,t,w,i); }

            { fe lam,num,ny,x3,t0,y3;
              fe_negate(&ny,&c_Gt[i].y);
              fe_sub(&num,&ny,&Pc.y); fe_mul(&lam,&num,&invd);
              fe_sqr(&x3,&lam); fe_sub(&x3,&x3,&Pc.x); fe_sub(&x3,&x3,&c_Gt[i].x);
              fe_sub(&t0,&Pc.x,&x3); fe_mul(&t0,&lam,&t0); fe_sub(&y3,&t0,&Pc.y);
              affine Q; Q.inf=0; Q.x=x3; Q.y=y3; check_point(&Q,t,w,-i); }
        }

        affine np; affine_add(&np,&Pc,&c_Sstride); Pc=np;
    }
}

static void scalar_n(scalar* r){ r->d[0]=0xBFD25E8CD0364141ULL;r->d[1]=0xBAAEDCE6AF48A03BULL;r->d[2]=0xFFFFFFFFFFFFFFFEULL;r->d[3]=0xFFFFFFFFFFFFFFFFULL; }
static int scalar_ge(const scalar* a,const scalar* b){ for(int i=3;i>=0;i--){ if(a->d[i]>b->d[i])return 1; if(a->d[i]<b->d[i])return 0;} return 1; }
static void scalar_sub(scalar* r,const scalar* a,const scalar* b){ u128 br=0; for(int i=0;i<4;i++){u128 c=(u128)a->d[i]-b->d[i]-br; r->d[i]=(uint64_t)c; br=(c>>64)&1;} }
static void scalar_reduce_n(scalar* s){ scalar n; scalar_n(&n); for(int k=0;k<4&&scalar_ge(s,&n);k++) scalar_sub(s,s,&n); }
static void scalar_to_b32(uint8_t b[32],const scalar* s){ for(int i=0;i<4;i++){uint64_t v=s->d[3-i]; for(int k=0;k<8;k++) b[i*8+k]=(uint8_t)(v>>(56-8*k)); } }

static void scalar_add_u128(scalar* r,const scalar* a,uint64_t lo,uint64_t hi){
    u128 s=(u128)a->d[0]+lo; r->d[0]=(uint64_t)s; u128 cy=s>>64;
    s=(u128)a->d[1]+hi+cy; r->d[1]=(uint64_t)s; cy=s>>64;
    for(int i=2;i<4;i++){ s=(u128)a->d[i]+cy; r->d[i]=(uint64_t)s; cy=s>>64; }
}
static void tohex(const uint8_t* b,int len,char* out){ static const char* H="0123456789abcdef"; for(int i=0;i<len;i++){out[i*2]=H[b[i]>>4];out[i*2+1]=H[b[i]&15];} out[len*2]=0; }

static void upload_tables(uint32_t nThreads){
    affine* Gt=(affine*)malloc(sizeof(affine)*W);
    Gt[0].inf=1; fe_set_zero(&Gt[0].x); fe_set_zero(&Gt[0].y);
    affine G; G.inf=0; secp_Gx(&G.x); secp_Gy(&G.y);
    Gt[1]=G;
    for (int i=2;i<W;i++) affine_add(&Gt[i],&Gt[i-1],&G);
    CK(cudaMemcpyToSymbol(c_Gt,Gt,sizeof(affine)*W));
    free(Gt);

    fe beta; secp_beta(&beta);
    CK(cudaMemcpyToSymbol(c_beta,&beta,sizeof(fe)));

    uint64_t span=2ULL*(W-1)+1;
    uint64_t stride=(uint64_t)nThreads*span;
    scalar ss; ss.d[0]=stride; ss.d[1]=ss.d[2]=ss.d[3]=0;
    affine S; scalar_mul_G(&S,&ss);
    CK(cudaMemcpyToSymbol(c_Sstride,&S,sizeof(affine)));
}

static scalar random_scalar(){
    std::random_device rd; std::mt19937_64 rng(((uint64_t)rd()<<32)^rd()^(uint64_t)time(0));
    scalar s; for(int i=0;i<4;i++) s.d[i]=rng();
    s.d[3]&=0x7FFFFFFFFFFFFFFFULL;
    scalar_reduce_n(&s);
    return s;
}

static int gpu_field_test(int count){
    std::mt19937_64 rng(0xC0FFEE);
    int n=count;
    FieldIn* hin=(FieldIn*)malloc(sizeof(FieldIn)*n);

    auto setfe=[&](fe* f,uint64_t a,uint64_t b,uint64_t c,uint64_t d){ f->d[0]=a;f->d[1]=b;f->d[2]=c;f->d[3]=d; };
    for (int i=0;i<n;i++){
        if (i==0){ setfe(&hin[i].a,1,0,0,0); setfe(&hin[i].b,1,0,0,0); }
        else if (i==1){ fe_p(&hin[i].a); hin[i].a.d[0]-=1; setfe(&hin[i].b,2,0,0,0); }
        else if (i==2){ fe_p(&hin[i].a); setfe(&hin[i].b,1,0,0,0); }
        else { for(int k=0;k<4;k++){ hin[i].a.d[k]=rng(); hin[i].b.d[k]=rng(); } }
        fe_cond_sub_p(&hin[i].a); fe_cond_sub_p(&hin[i].b);
        for (int k=0;k<8;k++) hin[i].t[k]=rng();
    }
    FieldIn* din; FieldOut* dout;
    CK(cudaMalloc(&din,sizeof(FieldIn)*n)); CK(cudaMalloc(&dout,sizeof(FieldOut)*n));
    CK(cudaMemcpy(din,hin,sizeof(FieldIn)*n,cudaMemcpyHostToDevice));
    k_field_test<<<(n+TPB-1)/TPB,TPB>>>(din,dout,n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    FieldOut* hout=(FieldOut*)malloc(sizeof(FieldOut)*n);
    CK(cudaMemcpy(hout,dout,sizeof(FieldOut)*n,cudaMemcpyDeviceToHost));
    long mism=0;
    for (int i=0;i<n;i++){
        fe mul,sqr,inv,add,sub,red; fe a=hin[i].a,b=hin[i].b;
        fe_mul(&mul,&a,&b); fe_sqr(&sqr,&a);
        if (!fe_is_zero(&a)) fe_inv(&inv,&a); else fe_set_zero(&inv);
        fe_add(&add,&a,&b); fe_sub(&sub,&a,&b);
        uint64_t t[8]; for(int k=0;k<8;k++) t[k]=hin[i].t[k]; fe_reduce512(&red,t);
        if(!fe_eq(&mul,&hout[i].mul)){mism++; if(mism<=5)fprintf(stderr,"mul mismatch i=%d\n",i);}
        if(!fe_eq(&sqr,&hout[i].sqr)){mism++; if(mism<=5)fprintf(stderr,"sqr mismatch i=%d\n",i);}
        if(!fe_eq(&inv,&hout[i].inv)){mism++; if(mism<=5)fprintf(stderr,"inv mismatch i=%d\n",i);}
        if(!fe_eq(&add,&hout[i].add)){mism++; if(mism<=5)fprintf(stderr,"add mismatch i=%d\n",i);}
        if(!fe_eq(&sub,&hout[i].sub)){mism++; if(mism<=5)fprintf(stderr,"sub mismatch i=%d\n",i);}
        if(!fe_eq(&red,&hout[i].red)){mism++; if(mism<=5)fprintf(stderr,"reduce mismatch i=%d\n",i);}
    }
    printf("gpufieldtest n=%d mismatches=%ld %s\n",n,mism,mism==0?"PASS":"FAIL");
    free(hin);free(hout);cudaFree(din);cudaFree(dout);
    return mism==0?0:1;
}

static int gpu_walkcheck(long N){
    upload_tables(1);
    scalar k0; k0.d[0]=0x123456789abcdef0ULL;k0.d[1]=0x0fedcba987654321ULL;k0.d[2]=0xdeadbeefcafebabeULL;k0.d[3]=0x0000000000000042ULL;
    unsigned long long* dm; CK(cudaMalloc(&dm,sizeof(unsigned long long)));
    k_walkcheck<<<1,1>>>(k0,N,dm);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long m; CK(cudaMemcpy(&m,dm,sizeof(m),cudaMemcpyDeviceToHost));
    printf("gpuwalkcheck N=%ld mismatches=%llu %s\n",N,m,m==0?"PASS":"FAIL");
    cudaFree(dm);
    return m==0?0:1;
}

static int gpu_endocheck(long N){
    upload_tables(1);
    scalar k0; k0.d[0]=0x0badf00dcafe0099ULL;k0.d[1]=0x0fedcba987654321ULL;k0.d[2]=0xdeadbeefcafebabeULL;k0.d[3]=0x0000000000000042ULL;
    unsigned long long* dm; CK(cudaMalloc(&dm,sizeof(unsigned long long)));
    k_endocheck<<<1,1>>>(k0,N,dm);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long m; CK(cudaMemcpy(&m,dm,sizeof(m),cudaMemcpyDeviceToHost));
    printf("gpuendocheck N=%ld variants=6 mismatches=%llu %s\n",N,m,m==0?"PASS":"FAIL");
    cudaFree(dm);
    return m==0?0:1;
}

static int gpu_derive(int n, uint64_t seed){
    std::mt19937_64 rng(seed);
    scalar* hk=(scalar*)malloc(sizeof(scalar)*n);
    uint8_t* hh=(uint8_t*)malloc(20*n);
    for (int i=0;i<n;i++){ for(int k=0;k<4;k++) hk[i].d[k]=rng(); hk[i].d[3]&=0x7FFFFFFFFFFFFFFFULL; scalar_reduce_n(&hk[i]); }
    scalar* dk; uint8_t* dh;
    CK(cudaMalloc(&dk,sizeof(scalar)*n)); CK(cudaMalloc(&dh,20*n));
    CK(cudaMemcpy(dk,hk,sizeof(scalar)*n,cudaMemcpyHostToDevice));
    k_derive<<<(n+TPB-1)/TPB,TPB>>>(dk,dh,n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hh,dh,20*n,cudaMemcpyDeviceToHost));
    for (int i=0;i<n;i++){
        uint8_t pb[32]; scalar_to_b32(pb,&hk[i]);
        char hpriv[65],hhex[41]; tohex(pb,32,hpriv); tohex(hh+i*20,20,hhex);
        std::string addr=p2wpkh_address(hh+i*20);
        printf("%s %s %s\n",hpriv,hhex,addr.c_str());
    }
    free(hk);free(hh);cudaFree(dk);cudaFree(dh);
    return 0;
}

// Pack the requested bech32 prefix (chars after "bc1q") into the leading bits of
// hash160, MSB-first, across TARGW 64-bit words. Up to 32 chars (5*32 = 160 bits).
static int prefix_to_target(const char* chars, uint64_t target[TARGW], uint64_t mask[TARGW], int* nbits){
    int m=(int)strlen(chars);
    int bits=5*m;
    if (bits>160){ fprintf(stderr,"prefix too long (max 32 chars after bc1q)\n"); return -1; }
    for (int k=0;k<TARGW;k++){ target[k]=0; mask[k]=0; }
    for (int i=0;i<m;i++){
        const char* pos=strchr(BECH32_CHARSET,chars[i]);
        if (!pos){ fprintf(stderr,"char '%c' not in bech32 charset\n",chars[i]); return -1; }
        uint64_t v=(uint64_t)(pos-BECH32_CHARSET);   // 0..31
        for (int j=0;j<5;j++){
            int b=5*i+j;                              // bit index from MSB of word 0
            int word=b/64, sh=63-(b%64);
            target[word] |= ((v>>(4-j))&1ULL) << sh;
            mask[word]   |= 1ULL << sh;
        }
    }
    *nbits=bits;
    return 0;
}

static int verify_and_print_hit(const scalar* launch_k0, uint32_t nThreads,
                                const Hit* h, const uint64_t target[TARGW], const uint64_t mask[TARGW],
                                const char* want_prefix){

    const uint64_t M=W-1, span=2ULL*M+1;
    u128 off=((u128)h->widx*nThreads + h->tid)*span + (u128)((int64_t)M + h->step);
    uint64_t lo=(uint64_t)off, hi=(uint64_t)(off>>64);
    scalar s; scalar_add_u128(&s,launch_k0,lo,hi);
    scalar_reduce_n(&s);

    // recover the actual private key for this candidate:  key = (-1)^neg * lambda^e * s (mod n)
    int e=h->variant>>1, neg=h->variant&1;
    scalar lambda; secp_lambda(&lambda);
    scalar key=s;
    for (int q=0;q<e;q++) scn_mul(&key,&key,&lambda);
    if (neg) scn_neg(&key,&key);

    uint8_t priv[32]; scalar_to_b32(priv,&key);
    affine P; scalar_mul_G(&P,&key);
    uint8_t h160[20]; pubkey_hash160(h160,&P);

    int hmatch=1; for(int k=0;k<20;k++) if(h160[k]!=h->h160[k]) hmatch=0;
    uint64_t w0=0,w1=0,w2=0;
    for(int i=0;i<8;i++)   w0=(w0<<8)|h160[i];
    for(int i=8;i<16;i++)  w1=(w1<<8)|h160[i];
    for(int i=16;i<20;i++) w2=(w2<<8)|h160[i];
    w2<<=32;
    int pmatch=((w0&mask[0])==target[0])&&((w1&mask[1])==target[1])&&((w2&mask[2])==target[2]);
    std::string a_w=p2wpkh_address(h160), a_p=p2pkh_address(h160), w=wif_compressed(priv);
    char hpriv[65],hhex[41]; tohex(priv,32,hpriv); tohex(h160,20,hhex);
    int ok = hmatch && pmatch && a_w.compare(0,strlen(want_prefix),want_prefix)==0;
    printf("HIT %s\n", ok?"VERIFIED":"REJECTED");
    printf("  PRIV %s\n", hpriv);
    printf("  HASH160 %s\n", hhex);
    printf("  P2WPKH %s\n", a_w.c_str());
    printf("  P2PKH %s\n", a_p.c_str());
    printf("  WIF %s\n", w.c_str());
    if (!ok){
        fprintf(stderr,"  HARD FAILURE: hmatch=%d pmatch=%d prefix_ok=%d\n",
                hmatch,pmatch,(int)(a_w.compare(0,strlen(want_prefix),want_prefix)==0));
    }
    return ok?0:1;
}

static int gpu_search(const char* full_prefix, int blocks, uint32_t windows, double maxsec){
    if (strncmp(full_prefix,"bc1q",4)!=0){ fprintf(stderr,"prefix must start with bc1q\n"); return 2; }
    const char* chars=full_prefix+4;
    uint64_t target[TARGW],mask[TARGW]; int nbits;
    if (prefix_to_target(chars,target,mask,&nbits)!=0) return 2;
    CK(cudaMemcpyToSymbol(c_target,target,sizeof(uint64_t)*TARGW));
    CK(cudaMemcpyToSymbol(c_mask,mask,sizeof(uint64_t)*TARGW));
    uint32_t nThreads=(uint32_t)blocks*TPB;
    upload_tables(nThreads);

    scalar k0=random_scalar();
    uint8_t k0b[32]; scalar_to_b32(k0b,&k0); char k0hex[65]; tohex(k0b,32,k0hex);
    printf("search prefix=%s nbits=%d threads=%u W=%d windows/launch=%u k0=%s\n",
           full_prefix,nbits,nThreads,W,windows,k0hex); fflush(stdout);

    unsigned int zero=0; CK(cudaMemcpyToSymbol(d_hitcount,&zero,sizeof(zero)));
    double total_keys=0; clock_t t0=clock();
    long launch=0; int rc=0;
    while (true){
        scalar launch_k0=k0;
        k_search<<<blocks,TPB>>>(k0,nThreads,windows);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint64_t span=2ULL*(W-1)+1;
        // each base point tests NENDO*NSIGN candidate addresses (point->pubkey->sha->ripemd->test),
        // so the address-throughput is that many times the base-point count.
        total_keys += (double)nThreads*span*windows*(NENDO*NSIGN);
        launch++;

        // k0 only advances over the contiguous base-point scalar space; the GLV
        // candidates live at lambda-multiples, not in this contiguous range.
        uint64_t adv=(uint64_t)nThreads*span*windows;
        scalar nk; scalar_add_u128(&nk,&k0,adv,0); k0=nk;

        unsigned int hc; CK(cudaMemcpyFromSymbol(&hc,d_hitcount,sizeof(hc)));
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC; if(el<=0)el=1e-9;
        if (hc>0){
            unsigned int got=hc>MAXHITS?MAXHITS:hc;
            Hit* hh=(Hit*)malloc(sizeof(Hit)*got);
            CK(cudaMemcpyFromSymbol(hh,d_hits,sizeof(Hit)*got));
            printf("launches=%ld keys=%.0f elapsed=%.2fs rate=%.3f Mkey/s hits=%u\n",
                   launch,total_keys,el,total_keys/el/1e6,hc);
            for (unsigned int j=0;j<got;j++)
                rc |= verify_and_print_hit(&launch_k0,nThreads,&hh[j],target,mask,full_prefix);
            free(hh);
            break;
        }
        if (maxsec>0 && el>=maxsec){
            printf("no hit: launches=%ld keys=%.0f elapsed=%.2fs rate=%.3f Mkey/s\n",
                   launch,total_keys,el,total_keys/el/1e6);
            break;
        }
        if (launch%5==0){ printf("  ...launch %ld keys=%.3e rate=%.3f Mkey/s\n",launch,total_keys,total_keys/el/1e6); fflush(stdout); }
    }
    return rc;
}

int main(int argc, char** argv){
    if (argc>=2 && strcmp(argv[1],"--gpufieldtest")==0){
        int c=argc>=3?atoi(argv[2]):512; return gpu_field_test(c);
    }
    if (argc>=2 && strcmp(argv[1],"--gpuwalkcheck")==0){
        long N=argc>=3?atol(argv[2]):4096; return gpu_walkcheck(N);
    }
    if (argc>=2 && strcmp(argv[1],"--gpuendocheck")==0){
        long N=argc>=3?atol(argv[2]):4096; return gpu_endocheck(N);
    }
    if (argc>=2 && strcmp(argv[1],"--gpuderive")==0){
        int n=argc>=3?atoi(argv[2]):512; uint64_t seed=argc>=4?strtoull(argv[3],0,10):42; return gpu_derive(n,seed);
    }
    if (argc>=2 && strcmp(argv[1],"--search")==0 && argc>=3){
        int blocks=2048; uint32_t windows=16; double maxsec=0;
        for (int i=3;i<argc;i++){
            if(!strcmp(argv[i],"--blocks")&&i+1<argc) blocks=atoi(argv[++i]);
            else if(!strcmp(argv[i],"--windows")&&i+1<argc) windows=(uint32_t)atol(argv[++i]);
            else if(!strcmp(argv[i],"--maxsec")&&i+1<argc) maxsec=atof(argv[++i]);
        }
        return gpu_search(argv[2],blocks,windows,maxsec);
    }
    fprintf(stderr,"usage: %s --gpufieldtest [n] | --gpuwalkcheck N | --gpuendocheck N | --gpuderive N [seed] | --search bc1q<prefix> [--blocks B --windows W --maxsec S]\n",argv[0]);
    return 2;
}
