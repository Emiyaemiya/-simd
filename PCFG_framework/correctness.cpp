#include "PCFG.h"
#include <chrono>
#include <fstream>

#include "md5.h"
#include <iomanip>
#include <vector>
using namespace std;
using namespace chrono;
//g++ correctness.cpp train.cpp guessing.cpp md5.cpp  -o main -O2
// 修改后的主函数
string md5_to_hex(const bit32 state[4]) {
    stringstream ss;
    for (int i = 0; i < 4; ++i) {
        uint32_t val = state[i];
        // 按小端顺序提取字节
        uint8_t bytes[4];
        bytes[0] = (val >> 0)  & 0xFF;  // 最低有效字节
        bytes[1] = (val >> 8)  & 0xFF;
        bytes[2] = (val >> 16) & 0xFF;
        bytes[3] = (val >> 24) & 0xFF;  // 最高有效字节
        
        // 按内存顺序拼接字节的十六进制表示
        for (int j = 0; j < 4; ++j) {
            ss << hex << setw(2) << setfill('0') << static_cast<int>(bytes[j]);
        }
    }
    return ss.str();
}
int main() {
    double time_hash = 0;  
    double time_guess = 0; 
    double time_train = 0; 
    PriorityQueue q;
    
  
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "System initialized" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    int history = 0;
    
    // 批量处理缓冲区
    const int SIMD_BATCH_SIZE = 4;
    vector<string> batch_buffer;
    bit32 batch_states[SIMD_BATCH_SIZE][4];

    while (!q.priority.empty()) {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        
        if (q.total_guesses - curr_num >= 100000) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
            
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n) {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash time: " << time_hash << " seconds" << endl;
                cout << "Train time: " << time_train << " seconds" << endl;
                break;
            }
        }

        if (curr_num > 1000000) {
            auto start_hash = system_clock::now();
            
            // 批量处理主逻辑
            int processed = 0;
            while (processed < q.guesses.size()) {
                // 填充批量缓冲区
                batch_buffer.clear();
                for (int i = 0; i < SIMD_BATCH_SIZE && processed < q.guesses.size(); ++i, ++processed) {
                    batch_buffer.push_back(q.guesses[processed]);
                }
                bit32 state[4];
                string expected[4];
                string out[4];
               
                // SIMD处理完整批次
                if (batch_buffer.size() == SIMD_BATCH_SIZE) {
                    string batch_inputs[SIMD_BATCH_SIZE];
                    bit32 ref[4][4];
                    for (int i=0;i<4;i++){
                        MD5Hash(batch_buffer[i],ref[i]);
                    }
                    copy(batch_buffer.begin(), batch_buffer.end(), batch_inputs);
                    MD5Hash_NEON(batch_inputs, batch_states);
                    for(int i=0;i<4;i++){
                        for(int j=0;j<4;j++){
                            if(ref[i][j]!=batch_states[i][j]){
                                cout<<"wrong parallel";
                                return 0;
                            }
                        }
                    }
                } 
                
                

               
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}