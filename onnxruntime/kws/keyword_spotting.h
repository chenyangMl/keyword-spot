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


#ifndef KWS_KEYWORD_SPOTTING_H_
#define KWS_KEYWORD_SPOTTING_H_

#include <memory>
#include <string>
#include <vector>
#include <iomanip>
#include <unordered_set>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "kws/utils.h"

namespace wekws {

    struct Token {
        int timeStep;  //  token time step
        int id;        //  token id of vocab
        float prob;    //  token prob
    };

    struct CtcPrefixBeamSearchOptions {
        int blank = 0;                // blank id of vocab list.
        int first_beam_size = 3;
        int second_beam_size = 3;
    };

    struct PrefixHash {
        size_t operator()(const std::vector<int>& prefix) const {
            size_t hash_code = 0;
            // here we use KB&DR hash code
            for (int id : prefix) {
                hash_code = id + 31 * hash_code;
            }
            return hash_code;
        }
    };

    struct PrefixScore {             // for one prefix.
        float s = 0.0;               // blank ending score
        float ns = 0.0;              // none blank ending score
        std::vector<Token> nodes;
        float total_score() const { return ns + s; }
    };

    // Define decoding type.
    typedef enum {
        DECODE_GREEDY_SEARCH=0,
        DECODE_PREFIX_BEAM_SEARCH=1,
    }DECODE_TYPE;

    class KeywordSpotting {
    public:
        explicit KeywordSpotting(const std::string &model_path, DECODE_TYPE decode_type, int model_type);

        void Reset();

        void reset_value();

        static void InitEngineThreads(int num_threads) {
            session_options_.SetIntraOpNumThreads(num_threads);
            session_options_.SetInterOpNumThreads(num_threads);
        }

        void Forward(const std::vector<std::vector<float>> &feats,
                     std::vector<std::vector<float>> *prob);

        // function to load vocab from token.txt
        void readToken(const std::string& tokenFile) ;

        // set keyword
        void setKeyWord(const std::string& keyWord);

        void decode_keywords(int stepT, std::vector<std::vector<float>>& probs);

        // decoding alignments to predict sequence using greedy search.
        void decode_with_greedy_search(int offset,  std::vector<std::vector<float>>& probs);

        // decoding alignments to predict sequence using prefix beam search.
        void decode_ctc_prefix_beam_search(int offset, const std::vector<float> &prob);

        // find keyword
        bool execute_detection();

        // Token is keyword or not.
        bool isKeyword(int index);

        //update current hypotheses from proposed extensions.
        void UpdateHypotheses(const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys);

        // maxpooling keywords
        std::vector<std::string> mmaxpooling_keywords;

        //time stemp reset
        void stepClear();

    private:
        // onnx runtime session
        static Ort::Env env_;
        static Ort::SessionOptions session_options_;
        std::shared_ptr<Ort::Session> session_ = nullptr;

        // model node names
        std::vector<const char *> in_names_;
        std::vector<const char *> out_names_;

        // meta info
        int cache_dim_ = 0;
        int cache_len_ = 0;
        int cache_4_ = 4;

        // cache info
        Ort::Value cache_ort_{nullptr};
        std::vector<float> cache_;

        // set mdoel type.
        int mmodel_type;

        //set decoder type.
        DECODE_TYPE mdecode_type;

        // vocab {token:index}
        std::unordered_map<std::string, int> mvocab;
        // keyword string
        std::string  mkey_word ;
        // keyword index set
        std::unordered_set<int> mkeyword_set;
        // keyword index list. handle same token in keyword.
        std::vector<int> mkeyword_token;

        // CTC alignments.
        std::vector<Token> alignments;
        // set of hypotheses with greed search.
        std::vector<Token> gd_cur_hyps;
        // set of hypotheses with prefix beam search.
        std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
        int total_frames=0;// frame offset, for absolute time

        //ctc prefix beam search
        const CtcPrefixBeamSearchOptions opts_={0, 3, 10};

        // silence time, 1s audio = 99frames melFbank. with default frame_shift(10ms) and frame_length(25ms).
        // Now we set silenceFrames = 3s * 99 = 297.
        //int mSilenceFrames = 297;

        //global Time step.
        int mGTimeStep = 0;
    };


}  // namespace wekws

#endif  // KWS_KEYWORD_SPOTTING_H_
