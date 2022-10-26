#include "bTree.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

struct _tagBTreeNode btree_node;
int BLOCK_SIZE = 4096;
int MAX_SUB_NUM = int(BLOCK_SIZE / sizeof(btree_node));
int DEGREE = int((MAX_SUB_NUM + 1) / 2);


int main()
{
    // 修改浮点输入/输出的默认格式
    std::cout << std::scientific;
    const int num_per_batch = 1000;
    std::string keys_file_path = "D:/lunwen/Learned-Indexes-master/data/linear.csv";
    auto time_limit = 100;
    double cumulative_operations = 0;
    double cumulative_insert_time = 0;
    double cumulative_lookup_time = 0;
    double cumulative_delete_time = 0;

    ifstream infile;
    infile.open(keys_file_path);  // csv的路径
    string s;
    vector<string> fields;       // 声明一个字符串向量
    while(getline(infile,s))
    {
        istringstream sin(s);    // 将整行字符串line读入到字符串流istringstream中
        string field;
        while (getline(sin, field, ','))     // 将字符串流sin中的字符读入到field字符串中，以逗号为分隔符
        {
            if (field.compare("") != 0){
                fields.push_back(field);       // 将刚刚读取的字符串添加到向量fields中
            }
        }
    }

    vector<Item> input_data;
    for (int i = 0;i < fields.size();i+=2){
        struct Item tempt;
        tempt.key = stof(fields[i]);
        tempt.value = stof(fields[i+1]);
        input_data.push_back(tempt);
    }

    int TOTAL_NUM = input_data.size();
    BTree b(DEGREE);

    int i = 0;
    int batch_no = 0;
    auto workload_start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        batch_no++;
        // Do inserts
        int num_actual_operations = std::min(num_per_batch, TOTAL_NUM - i);        // 判断剩余执行的操作数据量是否超出了总数据量
        int num_keys_after_batch = i + num_actual_operations;
        auto inserts_start_time = std::chrono::high_resolution_clock::now();
        for (int j = i; j < num_keys_after_batch; j++) {
            b.Insert(input_data[j]);
        }
        auto inserts_end_time = std::chrono::high_resolution_clock::now();
        double batch_insert_time = std::chrono::duration_cast<std::chrono::nanoseconds>(inserts_end_time - inserts_start_time).count();
        cumulative_insert_time += batch_insert_time;
        cumulative_operations += num_actual_operations;

        // Do lookups
        auto lookups_start_time = std::chrono::high_resolution_clock::now();
        for (int j = i; j < num_keys_after_batch; j++) {
            if (!b.Search(input_data[j])){
                cout<<"Not Present!"<<endl;
            };
        }
        auto lookups_end_time = std::chrono::high_resolution_clock::now();
        double batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lookups_end_time - lookups_start_time).count();
        cumulative_lookup_time += batch_lookup_time;


        // Do deletes
        auto deletes_start_time = std::chrono::high_resolution_clock::now();
        for (int j = i; j < num_keys_after_batch; j++) {
            b.Delete(input_data[j]);
        }
        auto deletes_end_time = std::chrono::high_resolution_clock::now();
        double batch_delete_time = std::chrono::duration_cast<std::chrono::nanoseconds>(deletes_end_time - deletes_start_time).count();
        cumulative_delete_time += batch_delete_time;

        // 更改后续开始执行操作的数据位置i
        i = num_keys_after_batch;

        // 判断workload的终止条件：将文件中的总数据全部执行完成 / 运行时间超过限制limit
        if (num_actual_operations < num_per_batch) {
            // 将文件中的总数据全部执行完成
            break;
        }

        // 运行时间超过限制limit
        double workload_elapsed_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - workload_start_time).count();
        if (workload_elapsed_time > time_limit * 1e9 * 60) {
            break;
        }
    }

    // 输出最终操作执行时间
    std::cout << "Cumulative stats: " << batch_no << " batches, "
              << cumulative_operations << " ops "

              << "\ncumulative ops time:\t"
              << cumulative_lookup_time / cumulative_operations * 1e-9
              << " lookups time,\t"
              << cumulative_insert_time / cumulative_operations * 1e-9
              << " inserts time,\t"
              << cumulative_delete_time / cumulative_operations * 1e-9
              << " deletes time"

              << std::endl;


}