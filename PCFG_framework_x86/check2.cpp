#include "md5.h"
int main(){
    string inputs[4] = {"hello", "hello", "hello", "hello"};
    bit32 states[4][4];
    MD5Hash_SSE(inputs, states);
}