// Copyright (c) 2017 Personal (Binbin Zhang)
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

#include "frontend/feature_pipeline.h"

#include <algorithm>
#include <utility>

namespace wenet {

    FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig &config)
            : config_(config),
              feature_dim_(config.num_bins),
              fbank_(config.num_bins, config.sample_rate, config.frame_length,
                     config.frame_shift),
              num_frames_(0),
              input_finished_(false) {}

    void FeaturePipeline::AcceptWaveform(const std::vector<float> &wav) {
        std::vector<std::vector<float>> feats;
        std::vector<float> waves;
        waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
        waves.insert(waves.end(), wav.begin(), wav.end());
        int num_frames = fbank_.Compute(waves, &feats); // feats.shape=(frames, mel_num_bins)

        if (config_.model_type==CTC_TYPE_MODEL){
            int left_context = config_.left_context, right_context = config_.right_context;

            // 处理CTC Loss的模型的特征输入，参考https://modelscope.cn/studios/thuduj12/KWS_Nihao_Xiaojing/file/view/master/stream_kws_ctc.py
            // 将mel_num_bins=80的音频数据处理成dim=400的数据
            std::vector<std::vector<float>> feats_pad;
            if(!feature_remained.empty()){
                feats_pad.insert(feats_pad.end(), feature_remained.begin(), feature_remained.end());
                feats_pad.insert(feats_pad.end(), feats.begin(), feats.end());
                feature_remained.clear(); // clear for updating later.
            }else{
                feats_pad = std::move(padFeatures(feats, left_context));
            }
            std::vector<std::vector<float>> feats_ctx = extractContext(feats_pad, left_context,
                                                                       right_context);

            // update feature remained, and feats
            int feature_remained_size = left_context + right_context;
            if (feature_remained_size <= feats.size()){
                int start_index = feats.size() - feature_remained_size;
                feature_remained.assign(feats.begin() + start_index, feats.end());
            }

            //对序列进行skip采样，降低重复计算。
//            int last_remainder = 0;
//            int remainder = (feats.size() + last_remainder) % this->config_.downsampling;
            // 对feats_ctx特征进行切片，按照step进行。
            std::vector<std::vector<float>> feats_down = slice(feats_ctx, 0, config_.downsampling);


            for (size_t i = 0; i < feats_down.size(); ++i) {
                feature_queue_.Push(std::move(feats_down[i]));
            }
        }else{
            for (size_t i = 0; i < feats.size(); ++i) {
                feature_queue_.Push(std::move(feats[i]));
            }
        }

        num_frames_ += num_frames;

        int left_samples = waves.size() - config_.frame_shift * num_frames;
        remained_wav_.resize(left_samples);
        std::copy(waves.begin() + config_.frame_shift * num_frames, waves.end(),
                  remained_wav_.begin());
        // We are still adding wave, notify input is not finished
        finish_condition_.notify_one();
    }

    void FeaturePipeline::AcceptWaveform(const std::vector<int16_t> &wav) {
        std::vector<float> float_wav(wav.size());
        for (size_t i = 0; i < wav.size(); i++) {
            float_wav[i] = static_cast<float>(wav[i]);
        }
        this->AcceptWaveform(float_wav);
    }

    void FeaturePipeline::set_input_finished() {
        CHECK(!input_finished_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            input_finished_ = true;
            feature_remained.clear();
        }
        finish_condition_.notify_one();
    }

    bool FeaturePipeline::ReadOne(std::vector<float> *feat) {
        if (!feature_queue_.Empty()) {
            *feat = std::move(feature_queue_.Pop());
            return true;
        } else {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!input_finished_) {
                // This will release the lock and wait for notify_one()
                // from AcceptWaveform() or set_input_finished()
                finish_condition_.wait(lock);
                if (!feature_queue_.Empty()) {
                    *feat = std::move(feature_queue_.Pop());
                    return true;
                }
            }
            CHECK(input_finished_);
            // Double check queue.empty, see issue#893 for detailed discussions.
            if (!feature_queue_.Empty()) {
                *feat = std::move(feature_queue_.Pop());
                return true;
            } else {
                return false;
            }
        }
    }

    bool FeaturePipeline::Read(int num_frames,
                               std::vector<std::vector<float>> *feats) {
        feats->clear();
        std::vector<float> feat;
        while (feats->size() < num_frames) {
            if (ReadOne(&feat)) {
                feats->push_back(std::move(feat));
            } else {
                return false;
            }
        }
        return true;
    }

    void FeaturePipeline::Reset() {
        input_finished_ = false;
        num_frames_ = 0;
        remained_wav_.clear();
        feature_queue_.Clear();
    }

    std::vector<std::vector<float>>
    FeaturePipeline::padFeatures(const std::vector<std::vector<float>> &feats, int leftContext) {
        // 获取特征矩阵的行数和列数
        size_t numRows = feats.size();
        size_t numCols = feats[0].size();

        // 计算填充后的特征矩阵的列数
        size_t paddedRows = numRows + leftContext;

        // 创建填充后的特征矩阵
        std::vector<std::vector<float>> paddedFeats(paddedRows, std::vector<float>(numCols));

        // 复制原始特征到填充后的特征矩阵中
        for (size_t i = 0; i < numRows; ++i) {
            // 复制原始特征到填充后的特征矩阵的右侧
            std::copy(feats[i].begin(), feats[i].end(), paddedFeats[i + leftContext].begin());

            // 使用最边缘的元素复制填充特征到填充后的特征矩阵的左侧
            for (int j = 0; j < leftContext; ++j) {
                std::copy(feats[0].begin(), feats[0].end(), paddedFeats[j].begin()); // 使用第一行的元素进行复制
            }
        }

        return paddedFeats;
    }


    std::vector<std::vector<float>> FeaturePipeline::extractContext(const std::vector<std::vector<float>> &
    featsPad, int leftContext, int rightContext) {
        int ctxFrm = featsPad.size() - (leftContext + rightContext);
        int ctxWin = leftContext + rightContext + 1;
        int ctxDim = featsPad[0].size() * ctxWin;

        std::vector<std::vector<float>> featsCtx(ctxFrm, std::vector<float>(ctxDim, 0.0));

        for (int i = 0; i < ctxFrm; ++i) {
            int start = i;
            int end = i + ctxWin;
            int index = 0;

            for (int j = start; j < end; ++j) {
                for (int k = 0; k < featsPad[j].size(); ++k) {
                    featsCtx[i][index] = featsPad[j][k];
                    ++index;
                }
            }
        }

        return featsCtx;
    }

    std::vector<std::vector<float>> FeaturePipeline::slice(const std::vector<std::vector<float>>& data, int start, int step) {
        std::vector<std::vector<float>> result;
        for (int i = start; i < data.size(); i += step) {
            result.push_back(data[i]);
        }
        return result;
    }
}  // namespace wenet
