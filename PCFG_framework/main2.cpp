#include "PCFG.h"
#include <chrono>
#include <fstream>

#include "md5.h"
#include <iomanip>
#include <vector>
using namespace std;
using namespace chrono;
/*
Guess time: 2.4737 seconds
Hash time: 3.4899 seconds
Train time: 30.6297 seconds

O1
Guess time: 0.219906 seconds
Hash time: 0.625426 seconds
Train time: 8.53487 seconds


O2
Guess time: 0.231175 seconds
Hash time: 0.635691 seconds
Train time: 8.17612 seconds

*/
/*
g++ main2.cpp train.cpp guessing.cpp md5.cpp -o main2 -mavx2
g++ main2.cpp train.cpp guessing.cpp md5.cpp -o O1main2 -O1 -mavx2
    g++ main2.cpp train.cpp guessing.cpp md5.cpp -o O2main2 -O2 -mavx2
*/
// 修改后的主函数
int main() {
    double time_hash = 0;  
    double time_guess = 0; 
    double time_train = 0; 
    PriorityQueue q;
    
  
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
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

                // SIMD处理完整批次
                if (batch_buffer.size() == SIMD_BATCH_SIZE) {
                    string batch_inputs[SIMD_BATCH_SIZE];
                    copy(batch_buffer.begin(), batch_buffer.end(), batch_inputs);
                    MD5Hash_SSE(batch_inputs, batch_states);
                // for (int i = 0; i < batch_buffer.size(); ++i) {
                //     cout << batch_buffer[i] << "\t";
                //     for (int j = 0; j < 4; ++j) {
                //         cout << hex << setw(8) << setfill('0') << batch_states[i][j];
                //     }
                // cout<<endl;} 
                // 处理剩余不足批次的情况
                }
                else {
                    for (const string& pw : batch_buffer) {
                        bit32 state[4];
                        MD5Hash(pw, state);
                        // cout<<pw<<"\t";
                        // for (int i1 = 0; i1 < 4; i1 += 1) {
                        //     cout << std::setw(8) << std::setfill('0') << hex << state[i1];
                        // }
                        // cout << endl;
                    }
                }

                // 这里可以添加结果输出逻辑
                
                //     cout << endl;}
                    
                // }//这里没有考虑不足批次的情况所以不足的部分输出有误，因为只使用了batch_states[i][j]
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
//g++ main2.cpp train.cpp guessing.cpp md5.cpp -o test2.exe -O1