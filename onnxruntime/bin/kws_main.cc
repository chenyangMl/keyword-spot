// Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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


#include <algorithm>
#include <string>

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "kws/keyword_spotting.h"
#include "utils/log.h"
#include <boost/filesystem.hpp>  //apt-get install libboost-all-dev


void read_pcm(const std::string& file_path, std::vector<float>& pcm_float){
    std::ifstream pcm_file(file_path, std::ios::binary | std::ios::ate);

    if (!pcm_file.is_open()) {
        std::cout << "Failed to open PCM file." << std::endl;
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


int main(int argc, char *argv[]) {

    std::string token_path;
    std::string key_word;
    wenet::MODEL_TYPE mode_type;
    if (argc > 2){
        mode_type = (wenet::MODEL_TYPE)std::stoi(argv[1]);
        if(mode_type==wenet::CTC_TYPE_MODEL){
            if (argc != 7) {
                LOG(FATAL) << "Usage: kws_main\n ./kws_main [solution_type, int] [num_bins, int] "
                << "[batch_size, int] [model_path, str] [wave_path,str] [key_word,str]"   ;
            }
            // Input Arguments.
            key_word = argv[6];
            token_path = "../../kws/tokens.txt";
        } else if (mode_type == wenet::MAXPOOLING_TYPE_MODEL){
            if (argc != 6) {
                LOG(FATAL) << "Usage: kws_main\n [solution_type, int] [num_bins, int] "
                <<"[batch_size, int] [model_path, str] [wave_path,str]" ;
            }
            token_path = "../../kws/maxpooling_keyword.txt";
        }
    }else{
        LOG(FATAL) << "Usage: kws_main\n [solution_type, int] [num_bins, int] "
                   <<"[batch_size, int] [model_path, str] [wave_path,str]" ;
    }

    // Input Arguments.
    const int num_bins = std::stoi(argv[2]);             // num_mel_bins in config.yaml. means dim of Fbank feature.
    const int batch_size = std::stoi(argv[3]);
    if(batch_size < 1){
        LOG(FATAL) << "batch_size should greater than 0, it's equal to " << batch_size << "now";
    }
    const std::string model_path = argv[4];
    const std::string wav_path = argv[5];


    boost::filesystem::path wavpath(wav_path);
    std::vector<float> wav;
    if (wavpath.extension() == ".wav"){
        // audio reader
        wenet::WavReader wav_reader(wav_path);
        int num_samples = wav_reader.num_samples();
        wav.assign(wav_reader.data(), wav_reader.data() + num_samples);
    }else if (wavpath.extension() == ".pcm"){
        read_pcm(wav_path, wav);
    }else{
        LOG(FATAL) << "Not support format = " << wavpath.extension();
    }

    // Setting config for handling waveform of audio, convert it to mel spectrogram of audio.
    // Only support CTC_TYPE_MODEL.
    wenet::FeaturePipelineConfig feature_config(num_bins, 16000, mode_type);
    wenet::FeaturePipeline feature_pipeline(feature_config);
    feature_pipeline.AcceptWaveform(wav);
    feature_pipeline.set_input_finished();

    wekws::KeywordSpotting spotter(model_path, wekws::DECODE_PREFIX_BEAM_SEARCH, mode_type);
    spotter.readToken(token_path);
    if(mode_type==1){
        // set keyword
        spotter.setKeyWord(key_word);
    }

    // Simulate streaming, detect batch by batch
    int offset = 0;
    while (true) {
        std::vector<std::vector<float>> feats;
        bool ok = feature_pipeline.Read(batch_size, &feats);
        std::vector<std::vector<float>> probs; //
        spotter.Forward(feats, &probs);

        if(mode_type==1){
            // Reach the end of feature pipeline
            spotter.decode_keywords(offset, probs); // feature_config.downsampling
            bool flag = spotter.execute_detection();
        }else{
            int flag = 0;
            float threshold = 0.8; // > threshold  means keyword activated. < threshold means not.
            for (int i = 0; i < probs.size(); i++) {
                std::cout << "frame " << offset + i << " prob";
                for (int j = 0; j < probs[i].size(); j++) { // size()=number of keywords.

                    std::cout << " " << probs[i][j];
                    if (probs[i][j] > threshold){
                        std::cout << " activated keyword: " << spotter.mmaxpooling_keywords[j] << " ";
                    }
                }
                std::cout << std::endl;
            }
        }

        if (!ok) break;
        offset += probs.size();
    }
    return 0;
}
