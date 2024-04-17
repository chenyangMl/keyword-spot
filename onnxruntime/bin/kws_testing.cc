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
// limitations under the License
//
//.


#include <algorithm>
#include <string>

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "kws/keyword_spotting.h"
#include <boost/filesystem.hpp>

int main(int argc, char* argv[]){

    // Input Arguments.
    const int num_bins = std::stoi(argv[2]);             // num_mel_bins in config.yaml. means dim of Fbank feature.
    const int batch_size = std::stoi(argv[3]);
    if(batch_size < 1){
        LOG(FATAL) << "batch_size should greater than 0, it's equal to " << batch_size << "now";
    }
    const std::string model_path = argv[4];
    const std::string key_word = argv[5];
    const std::string token_path = "../../kws/tokens.txt";
    const std::string test_dir = "/mnt3/datas/nlp/keywords_spot/datasets/turing_dataset/testset/zhiwatongxue-testset/zhiwatongxue-pos/adult";



    // Setting config for handling waveform of audio, convert it to mel spectrogram of audio.
    // Only support CTC_TYPE_MODEL.
    wenet::FeaturePipelineConfig feature_config(num_bins, 16000, wenet::CTC_TYPE_MODEL);
    wenet::FeaturePipeline feature_pipeline(feature_config);


    wekws::KeywordSpotting spotter(model_path, wekws::DECODE_PREFIX_BEAM_SEARCH, wenet::CTC_TYPE_MODEL);
    spotter.readToken(token_path);
    spotter.setKeyWord(key_word);

    int TP=0, FP=0, FN=0, TN=0;
    // iter directory.
    boost::filesystem::path directory(test_dir);
    for(boost::filesystem::directory_iterator it(directory); it != boost::filesystem::directory_iterator(); ++it){
        if (boost::filesystem::is_regular_file(it->path()) && it->path().extension() == ".wav") {
            // 读取 WAV 文件
            std::cout << "读取 WAV 文件：" << it->path().filename() << std::endl;
            // ...

            // audio reader
            wenet::WavReader wav_reader(wav_path);
            int num_samples = wav_reader.num_samples();
            std::vector<float> wav(wav_reader.data(), wav_reader.data() + num_samples);

            feature_pipeline.AcceptWaveform(wav);
            feature_pipeline.set_input_finished();

            // Simulate streaming, detect batch by batch
            int offset = 0;
            while (true) {
                std::vector<std::vector<float>> feats;
                bool ok = feature_pipeline.Read(batch_size, &feats);
                std::vector<std::vector<float>> probs; //
                spotter.Forward(feats, &probs);

                // Reach the end of feature pipeline
                spotter.decode_keywords(offset, probs); // feature_config.downsampling
                bool flag = spotter.execute_detection();
                if(flag==1){
                    TP += 1;
                }else{
                    FN +=1;
                }
                if (!ok) break;
                offset += probs.size();
            }
        }
    }

    // compute metric
    std::cout << "TP: " << TP << "FN: " << FN << std::endl;

    return  0;
};