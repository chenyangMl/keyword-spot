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


#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "kws/keyword_spotting.h"
#include "kws/utils.h"

using namespace wekws;

int main(int argc, char *argv[]) {

    // Input Arguments.
    const int num_bins = std::stoi(argv[1]);             // num_mel_bins in config.yaml. means dim of Fbank feature.
    const int batch_size = 2;      //固定batch_size测试
    if (batch_size < 1) {
        LOG(FATAL) << "batch_size should greater than 0, it's equal to " << batch_size << "now";
    }
    const std::string model_path = argv[2];
    const std::string key_word = argv[3];
    const std::string token_path = "../../kws/tokens.txt";
    const std::string test_dir = argv[4];
    const int interval = std::stoi(argv[5]); // 每次输入多少ms的音频数据

    // Setting config for handling waveform of audio, convert it to mel spectrogram of audio.
    // Only support CTC_TYPE_MODEL.
    wekws::KeywordSpotting spotter(model_path, wekws::DECODE_PREFIX_BEAM_SEARCH, 1);
    spotter.readToken(token_path);
    spotter.setKeyWord(key_word);

    std::vector<std::string> wavepath;
    // walk path, collection all wave file.
    boost::filesystem::path directory(test_dir);
    process_directory(directory, wavepath);

    wenet::FeaturePipelineConfig feature_config(num_bins, 16000, wenet::CTC_TYPE_MODEL);
    wenet::FeaturePipeline feature_pipeline(feature_config);
    feature_pipeline.set_input_finished();

    int TP = 0, FN = 0;
    std::vector<std::string> errorCases;
    for (const std::string wav_path: wavepath) {

        // audio reader
        wenet::WavReader wav_reader(wav_path);
        int num_samples = wav_reader.num_samples();
        std::vector<float> wav1(wav_reader.data(), wav_reader.data() + num_samples);

        int count = 0;
        //分段传入, 每100ms传入一次数据，100ms=100*(16000/1000)=1600nums, 100ms=1600*2=3200 bytes。
        std::vector<float> wav;
        spotter.reset_value();
        spotter.stepClear();
        int numBytes = 0;
        bool flag = false;
        // Simulate streaming, detect batch by batch
        while (!wav1.empty()){
            wav.push_back(wav1.front());
            wav1.erase(wav1.begin());
            count +=1;
            if (count < interval*16 && !wav1.empty() ) {
                continue;
            }else{
                count = 0;
            }
            numBytes += wav.size();
            feature_pipeline.AcceptWaveform(wav);
            wav.clear();

            while (true) {
                std::vector<std::vector<float>> feats;

                bool ok = feature_pipeline.Read(batch_size, &feats);
                std::vector<std::vector<float>> probs; //

                spotter.Forward(feats, &probs);
                std::cout << "feats.size= " << feats.size() << " probs.size=" << probs.size() << std::endl;
                // Reach the end of feature pipeline
                spotter.decode_keywords(probs); // feature_config.downsampling
                if (spotter.kwsInfo.state) flag = true;
                if (!ok) break;
            }
        }
        if (flag) {
            TP += 1;
            // find keyword in predicted sequence.
            std::cout << "YES ：" << wav_path << "\t" << numBytes << std::endl;
        } else {
            FN += 1;
            std::cout << "NO ：" << wav_path << "\t" << numBytes << std::endl;
            errorCases.push_back(wav_path);
        }
    }

    // compute metric
    std::cout << "TP: " << TP << " FN: " << FN << std::endl;
    // save bad case
    std::string badcasePath = "test.txt";
    writeVectorToFile(errorCases, badcasePath);

    return 0;
};