// Copyright (c) 2024 Yang Chen (cyang8050@163.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// reference code from:
//  https://github.com/wenet-e2e/wenet/blob/main/runtime/core/utils/utils.cc
//

#include "kws/utils.h"


namespace wekws {

    template <typename T>
    struct ValueComp {
        bool operator()(const std::pair<T, int32_t>& lhs,
                        const std::pair<T, int32_t>& rhs) const {
            return lhs.first > rhs.first ||
                   (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    // We refer the pytorch topk implementation
    // https://github.com/pytorch/pytorch/blob/master/caffe2/operators/top_k.cc
    template <typename T>
    void TopK(const std::vector<T>& data, int32_t k, std::vector<T>* values,
              std::vector<int>* indices) {
        std::vector<std::pair<T, int32_t>> heap_data;
        int n = data.size();
        for (int32_t i = 0; i < k && i < n; ++i) {
            heap_data.emplace_back(data[i], i);
        }
        std::priority_queue<std::pair<T, int32_t>, std::vector<std::pair<T, int32_t>>,
                ValueComp<T>>
        pq(ValueComp<T>(), std::move(heap_data));
        for (int32_t i = k; i < n; ++i) {
            if (pq.top().first < data[i]) {
                pq.pop();
                pq.emplace(data[i], i);
            }
        }

        values->resize(std::min(k, n));
        indices->resize(std::min(k, n));
        int32_t cur = values->size() - 1;
        while (!pq.empty()) {
            const auto& item = pq.top();
            (*values)[cur] = item.first;
            (*indices)[cur] = item.second;
            pq.pop();
            cur -= 1;
        }
    }

    template void TopK<float>(const std::vector<float>& data, int32_t k,
                              std::vector<float>* values,
                              std::vector<int>* indices);


    //读取PCM音频文件为vector
    void read_pcm(const std::string& file_path, std::vector<float>& pcm_float){
        std::ifstream pcm_file(file_path, std::ios::binary | std::ios::ate);

        if (!pcm_file.is_open()) {
            throw std::runtime_error("Failed to open file:" + file_path); // 抛出异常
        }

        // 获取文件大小
        std::streampos file_size = pcm_file.tellg();
        pcm_file.seekg(0, std::ios::beg);

        // 读取PCM数据
        std::vector<char> pcm_data(file_size);
        pcm_file.read(pcm_data.data(), file_size);
        pcm_file.close();

        // 将PCM数据转换为float数据
        const int16_t* pcm_data_ptr = reinterpret_cast<const int16_t*>(pcm_data.data());
        int sample_count = file_size / sizeof(int16_t);

        for (int i = 0; i < sample_count; ++i) {
            pcm_float.push_back(static_cast<float>(pcm_data_ptr[i]));
        }
    }

    void process_directory(const boost::filesystem::path &dirpath, std::vector<std::string> &wavePaths) {
        /*递归读取目录和子目录中的wav文件
         * */
        for (boost::filesystem::directory_iterator it(dirpath); it != boost::filesystem::directory_iterator(); ++it) {
            const boost::filesystem::path &path = it->path();
            if (boost::filesystem::is_regular_file(path) && path.extension() == ".wav") {
                // 读取 WAV 文件
                //std::cout << "读取 WAV 文件：" << path.string() << std::endl;
                wavePaths.push_back(path.string());
            } else if (boost::filesystem::is_directory(path)) {
                // 递归处理子目录
                process_directory(path, wavePaths);
            }
        }
    }

    void writeVectorToFile(const std::vector<std::string>& data, const std::string& filename) {
        std::ofstream file(filename);

        if (file.is_open()) {
            for (const auto& line : data) {
                file << line << std::endl;
            }
            file.close();
        } else {
            throw std::runtime_error("Failed to open file:" + filename); // 抛出异常
        }
    }

} // namespace wekws