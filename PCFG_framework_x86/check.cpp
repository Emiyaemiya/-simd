#include "md5.h"
#include <iomanip>
int main(){
    bit32 state0[4];
    bit32 state1[4];
    bit32 state2[4];
    bit32 state3[4];
    MD5Hash("hello", state0);
    MD5Hash("hello", state1);
    MD5Hash("hello", state2);
    MD5Hash("hello", state3);
}