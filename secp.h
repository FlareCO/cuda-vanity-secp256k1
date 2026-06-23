#ifndef SECP_H
#define SECP_H

#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

typedef unsigned __int128 u128;

typedef struct { uint64_t d[4]; } fe;

HD static inline void fe_p(fe* r) {
    r->d[0] = 0xFFFFFFFEFFFFFC2FULL;
    r->d[1] = 0xFFFFFFFFFFFFFFFFULL;
    r->d[2] = 0xFFFFFFFFFFFFFFFFULL;
    r->d[3] = 0xFFFFFFFFFFFFFFFFULL;
}

HD static inline void fe_set_zero(fe* r) { r->d[0]=r->d[1]=r->d[2]=r->d[3]=0; }
HD static inline void fe_set_one(fe* r)  { r->d[0]=1; r->d[1]=r->d[2]=r->d[3]=0; }
HD static inline int  fe_is_zero(const fe* a){ return (a->d[0]|a->d[1]|a->d[2]|a->d[3])==0; }
HD static inline void fe_copy(fe* r, const fe* a){ r->d[0]=a->d[0];r->d[1]=a->d[1];r->d[2]=a->d[2];r->d[3]=a->d[3]; }
HD static inline int  fe_eq(const fe* a, const fe* b){
    return a->d[0]==b->d[0]&&a->d[1]==b->d[1]&&a->d[2]==b->d[2]&&a->d[3]==b->d[3];
}

HD static inline int fe_ge_p(const fe* a){
    fe p; fe_p(&p);
    for (int i=3;i>=0;i--){
        if (a->d[i] > p.d[i]) return 1;
        if (a->d[i] < p.d[i]) return 0;
    }
    return 1;
}

HD static inline void fe_cond_sub_p(fe* a){
    if (fe_ge_p(a)){
        fe p; fe_p(&p);
        u128 borrow = 0;
        for (int i=0;i<4;i++){
            u128 cur = (u128)a->d[i] - p.d[i] - borrow;
            a->d[i] = (uint64_t)cur;
            borrow = (cur >> 64) & 1;
        }
    }
}

HD static inline void fe_reduce512(fe* r, uint64_t t[8]){
    const uint64_t C = 0x1000003D1ULL;

    for (int pass=0; pass<5; pass++){
        if ((t[4]|t[5]|t[6]|t[7]) == 0) break;
        uint64_t h[4] = { t[4], t[5], t[6], t[7] };
        t[4]=t[5]=t[6]=t[7]=0;

        uint64_t m[5];
        u128 carry = 0;
        for (int i=0;i<4;i++){ u128 cur=(u128)h[i]*C + carry; m[i]=(uint64_t)cur; carry=cur>>64; }
        m[4]=(uint64_t)carry;

        u128 cy = 0;
        for (int i=0;i<5;i++){ u128 s=(u128)t[i]+m[i]+cy; t[i]=(uint64_t)s; cy=s>>64; }
        int i=5;
        while (cy){ u128 s=(u128)t[i]+cy; t[i]=(uint64_t)s; cy=s>>64; i++; }
    }
    r->d[0]=t[0]; r->d[1]=t[1]; r->d[2]=t[2]; r->d[3]=t[3];
    fe_cond_sub_p(r);
    fe_cond_sub_p(r);
}

HD static inline void fe_add(fe* r, const fe* a, const fe* b){
    uint64_t t[8] = {0,0,0,0,0,0,0,0};
    u128 cy = 0;
    for (int i=0;i<4;i++){ u128 s=(u128)a->d[i]+b->d[i]+cy; t[i]=(uint64_t)s; cy=s>>64; }
    t[4]=(uint64_t)cy;
    fe_reduce512(r, t);
}

HD static inline void fe_sub(fe* r, const fe* a, const fe* b){
    u128 borrow = 0;
    uint64_t t[4];
    for (int i=0;i<4;i++){
        u128 cur=(u128)a->d[i]-b->d[i]-borrow;
        t[i]=(uint64_t)cur;
        borrow=(cur>>64)&1;
    }
    if (borrow){
        fe p; fe_p(&p);
        u128 cy=0;
        for (int i=0;i<4;i++){ u128 s=(u128)t[i]+p.d[i]+cy; t[i]=(uint64_t)s; cy=s>>64; }
    }
    r->d[0]=t[0]; r->d[1]=t[1]; r->d[2]=t[2]; r->d[3]=t[3];
}

HD static inline void fe_negate(fe* r, const fe* a){
    if (fe_is_zero(a)){ fe_set_zero(r); return; }
    fe p; fe_p(&p);
    u128 borrow=0; uint64_t t[4];
    for (int i=0;i<4;i++){ u128 cur=(u128)p.d[i]-a->d[i]-borrow; t[i]=(uint64_t)cur; borrow=(cur>>64)&1; }
    r->d[0]=t[0]; r->d[1]=t[1]; r->d[2]=t[2]; r->d[3]=t[3];
}

HD static inline void fe_mul(fe* r, const fe* a, const fe* b){
    uint64_t t[8] = {0,0,0,0,0,0,0,0};
    for (int i=0;i<4;i++){
        u128 carry=0;
        for (int j=0;j<4;j++){
            u128 cur=(u128)t[i+j] + (u128)a->d[i]*b->d[j] + carry;
            t[i+j]=(uint64_t)cur;
            carry=cur>>64;
        }
        t[i+4]+=(uint64_t)carry;
    }
    fe_reduce512(r, t);
}

HD static inline void fe_sqr(fe* r, const fe* a){ fe_mul(r, a, a); }

HD static inline void fe_mul_small(fe* r, const fe* a, uint64_t k){
    uint64_t t[8]={0,0,0,0,0,0,0,0};
    u128 carry=0;
    for (int i=0;i<4;i++){ u128 cur=(u128)a->d[i]*k+carry; t[i]=(uint64_t)cur; carry=cur>>64; }
    t[4]=(uint64_t)carry;
    fe_reduce512(r, t);
}

HD static inline void fe_inv(fe* r, const fe* a){

    const uint64_t e0=0xFFFFFFFEFFFFFC2DULL, e1=0xFFFFFFFFFFFFFFFFULL,
                   e2=0xFFFFFFFFFFFFFFFFULL, e3=0xFFFFFFFFFFFFFFFFULL;
    const uint64_t e[4]={e0,e1,e2,e3};
    fe result; fe_set_one(&result);
    fe base; fe_copy(&base, a);
    for (int limb=0; limb<4; limb++){
        uint64_t w=e[limb];
        for (int b=0;b<64;b++){
            if (w & 1ULL){ fe_mul(&result,&result,&base); }
            fe_mul(&base,&base,&base);
            w >>= 1;
        }
    }
    fe_copy(r,&result);
}

HD static inline void fe_set_b32(fe* r, const uint8_t b[32]){
    for (int i=0;i<4;i++){
        uint64_t v=0;
        const uint8_t* p=b + (3-i)*8;
        for (int k=0;k<8;k++) v=(v<<8)|p[k];
        r->d[i]=v;
    }
    fe_cond_sub_p(r);
}

HD static inline void fe_get_b32(uint8_t b[32], const fe* a){
    for (int i=0;i<4;i++){
        uint64_t v=a->d[3-i];
        for (int k=0;k<8;k++) b[i*8+k]=(uint8_t)(v>>(56-8*k));
    }
}

HD static inline int fe_is_odd(const fe* a){ return (int)(a->d[0] & 1ULL); }

HD static inline void secp_Gx(fe* r){
    r->d[0]=0x59F2815B16F81798ULL; r->d[1]=0x029BFCDB2DCE28D9ULL;
    r->d[2]=0x55A06295CE870B07ULL; r->d[3]=0x79BE667EF9DCBBACULL;
}
HD static inline void secp_Gy(fe* r){
    r->d[0]=0x9C47D08FFB10D4B8ULL; r->d[1]=0xFD17B448A6855419ULL;
    r->d[2]=0x5DA4FBFC0E1108A8ULL; r->d[3]=0x483ADA7726A3C465ULL;
}

typedef struct { fe x, y; int inf; } affine;
typedef struct { fe X, Y, Z; int inf; } jacob;

HD static inline void jac_set_inf(jacob* p){ p->inf=1; fe_set_zero(&p->X); fe_set_one(&p->Y); fe_set_zero(&p->Z); }

HD static inline void jac_from_affine(jacob* j, const affine* a){
    if (a->inf){ jac_set_inf(j); return; }
    j->inf=0; fe_copy(&j->X,&a->x); fe_copy(&j->Y,&a->y); fe_set_one(&j->Z);
}

HD static inline void jac_double(jacob* r, const jacob* p){
    if (p->inf || fe_is_zero(&p->Y)){ jac_set_inf(r); return; }
    fe A,B,C,D,E,F,t0,t1,X3,Y3,Z3;
    fe_sqr(&A,&p->X);
    fe_sqr(&B,&p->Y);
    fe_sqr(&C,&B);

    fe_add(&t0,&p->X,&B); fe_sqr(&t0,&t0); fe_sub(&t0,&t0,&A); fe_sub(&t0,&t0,&C);
    fe_add(&D,&t0,&t0);

    fe_add(&t1,&A,&A); fe_add(&E,&t1,&A);
    fe_sqr(&F,&E);

    fe_add(&t0,&D,&D); fe_sub(&X3,&F,&t0);

    fe_sub(&t0,&D,&X3); fe_mul(&t0,&E,&t0);
    fe_add(&t1,&C,&C); fe_add(&t1,&t1,&t1); fe_add(&t1,&t1,&t1);
    fe_sub(&Y3,&t0,&t1);

    fe_mul(&t0,&p->Y,&p->Z); fe_add(&Z3,&t0,&t0);
    r->inf=0; fe_copy(&r->X,&X3); fe_copy(&r->Y,&Y3); fe_copy(&r->Z,&Z3);
}

HD static inline void jac_add_affine(jacob* r, const jacob* p, const affine* q){
    if (q->inf){ *r=*p; return; }
    if (p->inf){ jac_from_affine(r,q); return; }
    fe Z1Z1,U2,S2,H,HH,I,J,rr,V,X3,Y3,Z3,t0,t1;
    fe_sqr(&Z1Z1,&p->Z);
    fe_mul(&U2,&q->x,&Z1Z1);
    fe_mul(&S2,&q->y,&p->Z); fe_mul(&S2,&S2,&Z1Z1);
    fe_sub(&H,&U2,&p->X);
    if (fe_is_zero(&H)){
        fe_sub(&t0,&S2,&p->Y);
        if (fe_is_zero(&t0)){ jac_double(r,p); return; }
        jac_set_inf(r); return;
    }
    fe_sqr(&HH,&H);
    fe_add(&I,&HH,&HH); fe_add(&I,&I,&I);
    fe_mul(&J,&H,&I);
    fe_sub(&rr,&S2,&p->Y); fe_add(&rr,&rr,&rr);
    fe_mul(&V,&p->X,&I);

    fe_sqr(&X3,&rr); fe_sub(&X3,&X3,&J); fe_add(&t0,&V,&V); fe_sub(&X3,&X3,&t0);

    fe_sub(&t0,&V,&X3); fe_mul(&t0,&rr,&t0);
    fe_mul(&t1,&p->Y,&J); fe_add(&t1,&t1,&t1);
    fe_sub(&Y3,&t0,&t1);

    fe_add(&t0,&p->Z,&H); fe_sqr(&t0,&t0); fe_sub(&t0,&t0,&Z1Z1); fe_sub(&Z3,&t0,&HH);
    r->inf=0; fe_copy(&r->X,&X3); fe_copy(&r->Y,&Y3); fe_copy(&r->Z,&Z3);
}

HD static inline void jac_to_affine(affine* a, const jacob* p){
    if (p->inf){ a->inf=1; fe_set_zero(&a->x); fe_set_zero(&a->y); return; }
    fe zinv, zinv2, zinv3;
    fe_inv(&zinv,&p->Z);
    fe_sqr(&zinv2,&zinv);
    fe_mul(&zinv3,&zinv2,&zinv);
    a->inf=0;
    fe_mul(&a->x,&p->X,&zinv2);
    fe_mul(&a->y,&p->Y,&zinv3);
}

HD static inline void affine_add(affine* r, const affine* p, const affine* q){
    if (p->inf){ *r=*q; return; }
    if (q->inf){ *r=*p; return; }
    fe num,den,inv,lam,x3,t;
    if (fe_eq(&p->x,&q->x)){
        fe ny; fe_negate(&ny,&q->y);
        if (fe_eq(&p->y,&ny)){ r->inf=1; fe_set_zero(&r->x); fe_set_zero(&r->y); return; }

        fe x2; fe_sqr(&x2,&p->x); fe_mul_small(&num,&x2,3);
        fe_add(&den,&p->y,&p->y);
    } else {
        fe_sub(&num,&q->y,&p->y);
        fe_sub(&den,&q->x,&p->x);
    }
    fe_inv(&inv,&den); fe_mul(&lam,&num,&inv);
    fe_sqr(&x3,&lam); fe_sub(&x3,&x3,&p->x); fe_sub(&x3,&x3,&q->x);
    fe_sub(&t,&p->x,&x3); fe_mul(&t,&lam,&t); fe_sub(&t,&t,&p->y);
    r->inf=0; fe_copy(&r->x,&x3); fe_copy(&r->y,&t);
}

typedef struct { uint64_t d[4]; } scalar;

HD static inline void scalar_mul_G(affine* out, const scalar* k){
    affine G; G.inf=0; secp_Gx(&G.x); secp_Gy(&G.y);
    jacob acc; jac_set_inf(&acc);

    for (int i=255;i>=0;i--){
        jac_double(&acc,&acc);
        uint64_t bit=(k->d[i>>6] >> (i & 63)) & 1ULL;
        if (bit) jac_add_affine(&acc,&acc,&G);
    }
    jac_to_affine(out,&acc);
}

HD static inline void scalar_add_u64(scalar* r, const scalar* a, uint64_t v){
    u128 s=(u128)a->d[0]+v; r->d[0]=(uint64_t)s; u128 cy=s>>64;
    for (int i=1;i<4;i++){ s=(u128)a->d[i]+cy; r->d[i]=(uint64_t)s; cy=s>>64; }
}

HD static inline void scalar_sub_u64(scalar* r, const scalar* a, uint64_t v){
    u128 s=(u128)a->d[0]-v; r->d[0]=(uint64_t)s; uint64_t br=(uint64_t)((s>>64)&1);
    for (int i=1;i<4;i++){ u128 c=(u128)a->d[i]-br; r->d[i]=(uint64_t)c; br=(uint64_t)((c>>64)&1); }
}

HD static inline uint32_t rotr32(uint32_t x, int n){ return (x>>n)|(x<<(32-n)); }

HD static inline void sha256(const uint8_t* msg, int len, uint8_t out[32]){
    const uint32_t K[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

    uint8_t buf[128]; int total=len;
    for (int i=0;i<len;i++) buf[i]=msg[i];
    buf[len]=0x80;
    int padded = ((len+8)/64 + 1)*64;
    for (int i=len+1;i<padded;i++) buf[i]=0;
    uint64_t bits=(uint64_t)total*8;
    for (int i=0;i<8;i++) buf[padded-1-i]=(uint8_t)(bits>>(8*i));
    for (int off=0; off<padded; off+=64){
        uint32_t w[64];
        for (int i=0;i<16;i++){
            w[i]=((uint32_t)buf[off+i*4]<<24)|((uint32_t)buf[off+i*4+1]<<16)|
                 ((uint32_t)buf[off+i*4+2]<<8)|((uint32_t)buf[off+i*4+3]);
        }
        for (int i=16;i<64;i++){
            uint32_t s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i=0;i<64;i++){
            uint32_t S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
            uint32_t ch=(e&f)^((~e)&g);
            uint32_t t1=hh+S1+ch+K[i]+w[i];
            uint32_t S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i=0;i<8;i++){ out[i*4]=(uint8_t)(h[i]>>24); out[i*4+1]=(uint8_t)(h[i]>>16); out[i*4+2]=(uint8_t)(h[i]>>8); out[i*4+3]=(uint8_t)h[i]; }
}

HD static inline uint32_t rol32(uint32_t x, int n){ return (x<<n)|(x>>(32-n)); }

HD static inline void ripemd160(const uint8_t* msg, int len, uint8_t out[20]){

    const int rl[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int rr[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int sl[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const int sr[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    const uint32_t KL[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    const uint32_t KR[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

    uint32_t h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
    uint8_t buf[128]; int total=len;
    for (int i=0;i<len;i++) buf[i]=msg[i];
    buf[len]=0x80;
    int padded=((len+8)/64 + 1)*64;
    for (int i=len+1;i<padded;i++) buf[i]=0;
    uint64_t bits=(uint64_t)total*8;
    for (int i=0;i<8;i++) buf[padded-8+i]=(uint8_t)(bits>>(8*i));
    for (int off=0; off<padded; off+=64){
        uint32_t X[16];
        for (int i=0;i<16;i++){
            X[i]=((uint32_t)buf[off+i*4])|((uint32_t)buf[off+i*4+1]<<8)|
                 ((uint32_t)buf[off+i*4+2]<<16)|((uint32_t)buf[off+i*4+3]<<24);
        }
        uint32_t al=h0,bl=h1,cl=h2,dl=h3,el=h4;
        uint32_t ar=h0,br=h1,cr=h2,dr=h3,er=h4;
        for (int j=0;j<80;j++){
            int rnd=j/16;
            uint32_t f,t;

            if (rnd==0) f=bl^cl^dl;
            else if (rnd==1) f=(bl&cl)|((~bl)&dl);
            else if (rnd==2) f=(bl|(~cl))^dl;
            else if (rnd==3) f=(bl&dl)|(cl&(~dl));
            else f=bl^(cl|(~dl));
            t=rol32(al+f+X[rl[j]]+KL[rnd], sl[j])+el;
            al=el; el=dl; dl=rol32(cl,10); cl=bl; bl=t;

            int rndr=4-rnd;
            if (rndr==0) f=br^cr^dr;
            else if (rndr==1) f=(br&cr)|((~br)&dr);
            else if (rndr==2) f=(br|(~cr))^dr;
            else if (rndr==3) f=(br&dr)|(cr&(~dr));
            else f=br^(cr|(~dr));
            t=rol32(ar+f+X[rr[j]]+KR[rnd], sr[j])+er;
            ar=er; er=dr; dr=rol32(cr,10); cr=br; br=t;
        }
        uint32_t tt=h1+cl+dr;
        h1=h2+dl+er; h2=h3+el+ar; h3=h4+al+br; h4=h0+bl+cr; h0=tt;
    }
    uint32_t hh[5]={h0,h1,h2,h3,h4};
    for (int i=0;i<5;i++){ out[i*4]=(uint8_t)hh[i]; out[i*4+1]=(uint8_t)(hh[i]>>8); out[i*4+2]=(uint8_t)(hh[i]>>16); out[i*4+3]=(uint8_t)(hh[i]>>24); }
}

HD static inline void hash160(const uint8_t* data, int len, uint8_t out[20]){
    uint8_t sh[32];
    sha256(data,len,sh);
    ripemd160(sh,32,out);
}

#define SHA_S0(x) (rotr32(x,2)^rotr32(x,13)^rotr32(x,22))
#define SHA_S1(x) (rotr32(x,6)^rotr32(x,11)^rotr32(x,25))
#define SHA_s0(x) (rotr32(x,7)^rotr32(x,18)^((x)>>3))
#define SHA_s1(x) (rotr32(x,17)^rotr32(x,19)^((x)>>10))

HD static inline void sha256_33(const uint8_t in[33], uint32_t out[8]){
    const uint32_t K[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[16];
    #define BE(p) (((uint32_t)(p)[0]<<24)|((uint32_t)(p)[1]<<16)|((uint32_t)(p)[2]<<8)|((uint32_t)(p)[3]))
    w[0]=BE(in+0); w[1]=BE(in+4); w[2]=BE(in+8);  w[3]=BE(in+12);
    w[4]=BE(in+16);w[5]=BE(in+20);w[6]=BE(in+24); w[7]=BE(in+28);
    w[8]=((uint32_t)in[32]<<24)|0x00800000u;
    w[9]=0;w[10]=0;w[11]=0;w[12]=0;w[13]=0;w[14]=0;
    w[15]=264;
    #undef BE
    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,
             e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma unroll
    for (int t=0;t<64;t++){
        uint32_t wt;
        if (t<16) wt=w[t];
        else {
            wt = w[t&15] + SHA_s0(w[(t+1)&15]) + w[(t+9)&15] + SHA_s1(w[(t+14)&15]);
            w[t&15]=wt;
        }
        uint32_t t1=h+SHA_S1(e)+((e&f)^((~e)&g))+K[t]+wt;
        uint32_t t2=SHA_S0(a)+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    out[0]=0x6a09e667+a; out[1]=0xbb67ae85+b; out[2]=0x3c6ef372+c; out[3]=0xa54ff53a+d;
    out[4]=0x510e527f+e; out[5]=0x9b05688c+f; out[6]=0x1f83d9ab+g; out[7]=0x5be0cd19+h;
}

HD static inline void ripemd160_32(const uint32_t sh[8], uint8_t out[20]){
    const int rl[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int rr[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int sl[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const int sr[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    const uint32_t KL[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    const uint32_t KR[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

    uint32_t X[16];
    #pragma unroll
    for (int i=0;i<8;i++){
        uint32_t v=sh[i];
        X[i]=((v>>24)&0xff)|(((v>>16)&0xff)<<8)|(((v>>8)&0xff)<<16)|((v&0xff)<<24);
    }
    X[8]=0x00000080u; X[9]=0;X[10]=0;X[11]=0;X[12]=0;X[13]=0;
    X[14]=256u; X[15]=0;
    uint32_t h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
    uint32_t al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint32_t ar=h0,br=h1,cr=h2,dr=h3,er=h4;
    #pragma unroll
    for (int j=0;j<80;j++){
        int rnd=j/16; uint32_t f,t;
        if (rnd==0) f=bl^cl^dl; else if (rnd==1) f=(bl&cl)|((~bl)&dl);
        else if (rnd==2) f=(bl|(~cl))^dl; else if (rnd==3) f=(bl&dl)|(cl&(~dl)); else f=bl^(cl|(~dl));
        t=rol32(al+f+X[rl[j]]+KL[rnd], sl[j])+el;
        al=el; el=dl; dl=rol32(cl,10); cl=bl; bl=t;
        int rndr=4-rnd;
        if (rndr==0) f=br^cr^dr; else if (rndr==1) f=(br&cr)|((~br)&dr);
        else if (rndr==2) f=(br|(~cr))^dr; else if (rndr==3) f=(br&dr)|(cr&(~dr)); else f=br^(cr|(~dr));
        t=rol32(ar+f+X[rr[j]]+KR[rnd], sr[j])+er;
        ar=er; er=dr; dr=rol32(cr,10); cr=br; br=t;
    }
    uint32_t tt=h1+cl+dr;
    h1=h2+dl+er; h2=h3+el+ar; h3=h4+al+br; h4=h0+bl+cr; h0=tt;
    uint32_t hh[5]={h0,h1,h2,h3,h4};
    #pragma unroll
    for (int i=0;i<5;i++){ out[i*4]=(uint8_t)hh[i]; out[i*4+1]=(uint8_t)(hh[i]>>8); out[i*4+2]=(uint8_t)(hh[i]>>16); out[i*4+3]=(uint8_t)(hh[i]>>24); }
}

HD static inline void pubkey_compressed(uint8_t out[33], const affine* p){
    out[0]= fe_is_odd(&p->y) ? 0x03 : 0x02;
    fe_get_b32(out+1,&p->x);
}

HD static inline void pubkey_hash160(uint8_t out[20], const affine* p){
    uint8_t pk[33];
    pk[0]= fe_is_odd(&p->y) ? 0x03 : 0x02;
    #pragma unroll
    for (int i=0;i<4;i++){
        uint64_t v=p->x.d[3-i];
        #pragma unroll
        for (int k=0;k<8;k++) pk[1+i*8+k]=(uint8_t)(v>>(56-8*k));
    }
    uint32_t sh[8];
    sha256_33(pk,sh);
    ripemd160_32(sh,out);
}

#endif
