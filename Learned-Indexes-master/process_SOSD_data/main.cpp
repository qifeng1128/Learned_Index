#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <chrono>
#include <algorithm>

static uint64_t timing(std::function<void()> fn) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
}

template <typename T>
static std::vector<T> load_data(const std::string& filename,
                                bool print = true) {
  std::vector<T> data;
  const uint64_t ns = timing([&] {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "unable to open " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    data.resize(size);
    // Read values.
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    in.close();
  });
  const uint64_t ms = ns / 1e6;

  if (print) {
    std::cout << "read " << data.size() << " values from " << filename << " in "
              << ms << " ms (" << static_cast<double>(data.size()) / 1000 / ms
              << " M values/s)" << std::endl;
  }

  return data;
}

static void fail(const std::string& message) {
    std::cerr << message << std::endl;
    exit(EXIT_FAILURE);
}

template <typename T>
static bool is_unique(const std::vector<T>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] == data[i - 1]) return false;
    }
    return true;
}

int main() {
    bool unique_keys_;

    std::string data_filename_ = "D:/lunwen/test_SOSD/data/fb_200M_uint64";
    // Load data
    std::vector<uint64_t> keys = load_data<uint64_t>(data_filename_);

    if (!is_sorted(keys.begin(), keys.end()))
        fail("keys have to be sorted");
    // Check whether keys are unique.
    unique_keys_ = is_unique(keys);
    if (unique_keys_)
        std::cout << "data is unique" << std::endl;
    else
        std::cout << "data contains duplicates" << std::endl;

    // 去除重复数值
    keys.erase( unique( keys.begin(), keys.end() ), keys.end() );

    // 将数据写入文件中
    int size = 1000000;
    if (keys.size() < size){
        size = keys.size();
    }

    std::ofstream outFile;
    outFile.open("D:/lunwen/Learned-Indexes-master/data/fb_200M_uint64.csv", std::ios::out);   // 打开模式可省略
    for(int i = 0; i < size; i++){
        outFile << keys[i] << ',' << i << std::endl;
    }
    outFile.close();


    return 0;
}