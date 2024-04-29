// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <queue>
#include <boost/filesystem.hpp>  //apt-get install libboost-all-dev
#include <stdexcept>

namespace wekws {

    template <typename T>
    void TopK(const std::vector<T>& data, int32_t k, std::vector<T>* values,
              std::vector<int>* indices);

    void read_pcm(const std::string& file_path, std::vector<float>& pcm_float);

    void process_directory(const boost::filesystem::path &dirpath, std::vector<std::string> &wavePaths);

    void writeVectorToFile(const std::vector<std::string>& data, const std::string& filename);
}  // namespace wekws

#endif  // UTILS_UTILS_H_
