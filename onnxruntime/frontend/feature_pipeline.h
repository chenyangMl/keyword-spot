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

#ifndef FRONTEND_FEATURE_PIPELINE_H_
#define FRONTEND_FEATURE_PIPELINE_H_

#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "frontend/fbank.h"
#include "utils/log.h"
#include "utils/blocking_queue.h"

namespace wenet {

    typedef enum {
        MAXPOOLING_TYPE_MODEL=0,
        CTC_TYPE_MODEL=1
    }MODEL_TYPE;

    struct FeaturePipelineConfig {
        int num_bins;
        int sample_rate;
        int frame_length;
        int frame_shift;
        int left_context;
        int right_context;
        int downsampling;
        MODEL_TYPE model_type; // 1:ctc 0: max-pooling

        FeaturePipelineConfig(int num_bins, int sample_rate, MODEL_TYPE model_type)
                : num_bins(num_bins),                  // 80 dim fbank. feature dim of mel-spectrogram.
                  sample_rate(sample_rate),           // 16k sample rate of audio
                  model_type(model_type) {
            frame_length = sample_rate / 1000 * 25;  // frame length 25ms, window_size
            frame_shift = sample_rate / 1000 * 10;   // frame shift 10ms, window_shift
            left_context = 2;                       // context_expansion_conf in config.yaml.
            right_context = 2;
            downsampling = 3;
        }

        void Info() const {
            LOG(INFO) << "feature pipeline config"
                      << " num_bins " << num_bins << " frame_length " << frame_length
                      << " frame_shift " << frame_shift;
        }
    };

// Typically, FeaturePipeline is used in two threads: one thread A calls
// AcceptWaveform() to add raw wav data and set_input_finished() to notice
// the end of input wav, another thread B (decoder thread) calls Read() to
// consume features.So a BlockingQueue is used to make this class thread safe.

// The Read() is designed as a blocking method when there is no feature
// in feature_queue_ and the input is not finished.

    class FeaturePipeline {
    public:
        explicit FeaturePipeline(const FeaturePipelineConfig &config);

        // The feature extraction is done in AcceptWaveform().
        void AcceptWaveform(const std::vector<float> &wav);

        void AcceptWaveform(const std::vector<int16_t> &wav);

        // Current extracted frames number.
        int num_frames() const { return num_frames_; }

        int feature_dim() const { return feature_dim_; }

        const FeaturePipelineConfig &config() const { return config_; }

        // The caller should call this method when speech input is end.
        // Never call AcceptWaveform() after calling set_input_finished() !
        void set_input_finished();

        bool input_finished() const { return input_finished_; }

        // Return False if input is finished and no feature could be read.
        // Return True if a feature is read.
        // This function is a blocking method. It will block the thread when
        // there is no feature in feature_queue_ and the input is not finished.
        bool ReadOne(std::vector<float> *feat);

        // Read #num_frames frame features.
        // Return False if less then #num_frames features are read and the
        // input is finished.
        // Return True if #num_frames features are read.
        // This function is a blocking method when there is no feature
        // in feature_queue_ and the input is not finished.
        bool Read(int num_frames, std::vector<std::vector<float>> *feats);

        void Reset();

        bool IsLastFrame(int frame) const {
            return input_finished_ && (frame == num_frames_ - 1);
        }

        int NumQueuedFrames() const { return feature_queue_.Size(); }

        std::vector<std::vector<float>> padFeatures(const std::vector<std::vector<float>> &feats, int leftContext);

        std::vector<std::vector<float>> extractContext(const std::vector<std::vector<float>> &
        featsPad, int leftContext, int rightContext);

        std::vector<std::vector<float>> slice(const std::vector<std::vector<float>> &data, int start, int step);

    private:
        const FeaturePipelineConfig &config_;
        int feature_dim_;
        Fbank fbank_;

        BlockingQueue<std::vector<float>> feature_queue_;
        int num_frames_;
        bool input_finished_;

        // The feature extraction is done in AcceptWaveform().
        // This wavefrom sample points are consumed by frame size.
        // The residual wavefrom sample points after framing are
        // kept to be used in next AcceptWaveform() calling.
        std::vector<float> remained_wav_;

        // Used to block the Read when there is no feature in feature_queue_
        // and the input is not finished.
        mutable std::mutex mutex_;
        std::condition_variable finish_condition_;

        std::vector<std::vector<float>> feature_remained;

    };

}  // namespace wenet

#endif  // FRONTEND_FEATURE_PIPELINE_H_
