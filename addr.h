#ifndef ADDR_H
#define ADDR_H

#include <stdint.h>
#include <string>
#include <vector>
#include "secp.h"

static const char* BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

static inline uint32_t bech32_polymod(const std::vector<uint8_t>& values){
    static const uint32_t GEN[5]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
    uint32_t chk=1;
    for (uint8_t v : values){
        uint8_t b=chk>>25;
        chk=((chk&0x1ffffff)<<5)^v;
        for (int i=0;i<5;i++) if ((b>>i)&1) chk^=GEN[i];
    }
    return chk;
}

static inline std::vector<uint8_t> bech32_hrp_expand(const std::string& hrp){
    std::vector<uint8_t> r;
    for (char c : hrp) r.push_back((uint8_t)c >> 5);
    r.push_back(0);
    for (char c : hrp) r.push_back((uint8_t)c & 31);
    return r;
}

static inline std::vector<uint8_t> bech32_checksum(const std::string& hrp, const std::vector<uint8_t>& data){
    std::vector<uint8_t> values=bech32_hrp_expand(hrp);
    values.insert(values.end(),data.begin(),data.end());
    for (int i=0;i<6;i++) values.push_back(0);
    uint32_t mod=bech32_polymod(values)^1;
    std::vector<uint8_t> ret;
    for (int i=0;i<6;i++) ret.push_back((mod>>(5*(5-i)))&31);
    return ret;
}

static inline std::string bech32_encode(const std::string& hrp, const std::vector<uint8_t>& data){
    std::vector<uint8_t> combined=data;
    std::vector<uint8_t> chk=bech32_checksum(hrp,data);
    combined.insert(combined.end(),chk.begin(),chk.end());
    std::string ret=hrp+"1";
    for (uint8_t d : combined) ret+=BECH32_CHARSET[d];
    return ret;
}

static inline std::vector<uint8_t> convertbits_8to5(const uint8_t* in, int inlen){
    std::vector<uint8_t> out;
    int acc=0, bits=0;
    for (int i=0;i<inlen;i++){
        acc=(acc<<8)|in[i];
        bits+=8;
        while (bits>=5){ bits-=5; out.push_back((acc>>bits)&31); }
    }
    if (bits>0) out.push_back((acc<<(5-bits))&31);
    return out;
}

static inline std::string p2wpkh_address(const uint8_t h160[20], const std::string& hrp="bc"){
    std::vector<uint8_t> data;
    data.push_back(0);
    std::vector<uint8_t> prog=convertbits_8to5(h160,20);
    data.insert(data.end(),prog.begin(),prog.end());
    return bech32_encode(hrp,data);
}

static inline std::string base58_encode(const uint8_t* data, int len){
    static const char* B58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    int zeros=0; while (zeros<len && data[zeros]==0) zeros++;
    std::vector<uint8_t> b(data,data+len);
    std::string result;
    int start=zeros;
    while (start<len){
        int rem=0;
        for (int i=start;i<len;i++){
            int acc=(rem<<8)|b[i];
            b[i]=(uint8_t)(acc/58);
            rem=acc%58;
        }
        result+=B58[rem];
        while (start<len && b[start]==0) start++;
    }
    for (int i=0;i<zeros;i++) result+='1';

    std::string out(result.rbegin(),result.rend());
    return out;
}

static inline void dsha256(const uint8_t* d, int len, uint8_t out[32]){
    uint8_t t[32]; sha256(d,len,t); sha256(t,32,out);
}

static inline std::string base58check(const uint8_t* payload, int len){
    std::vector<uint8_t> buf(payload,payload+len);
    uint8_t chk[32]; dsha256(payload,len,chk);
    buf.insert(buf.end(),chk,chk+4);
    return base58_encode(buf.data(),(int)buf.size());
}

static inline std::string p2pkh_address(const uint8_t h160[20]){
    uint8_t payload[21]; payload[0]=0x00;
    for (int i=0;i<20;i++) payload[i+1]=h160[i];
    return base58check(payload,21);
}

static inline std::string wif_compressed(const uint8_t priv[32]){
    uint8_t payload[34]; payload[0]=0x80;
    for (int i=0;i<32;i++) payload[i+1]=priv[i];
    payload[33]=0x01;
    return base58check(payload,34);
}

#endif
