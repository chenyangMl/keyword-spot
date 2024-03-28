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


#include "kws/keyword_spotting.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace wekws {

    Ort::Env KeywordSpotting::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
    Ort::SessionOptions KeywordSpotting::session_options_ = Ort::SessionOptions();

    static void print_vector(const std::vector<int> &arr) {
        if (!arr.empty()) {
            std::cout << "prefix: ";
            for (auto it: arr) {
                std::cout << it << ",";
            }
            std::cout << std::endl;
        }
    }

    static bool PrefixScoreCompare(
            const std::pair<std::vector<int>, PrefixScore> &a,
            const std::pair<std::vector<int>, PrefixScore> &b) {
        return a.second.total_score() > b.second.total_score();
    }

    KeywordSpotting::KeywordSpotting(const std::string &model_path, DECODE_TYPE decode_type, int model_type) {
        // 0. set decode type from {DECODE_GREEDY_SEARCH, DECODE_PREFIX_BEAM_SEARCH}
        mdecode_type = decode_type;
        mmodel_type = model_type;

        // 1. Load onnx runtime sessions
        session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                                  session_options_);
        // 2. Model info. Information can be view from netron.
        // pip install netron. netron [model_path]
        in_names_ = {"input", "cache"};
        out_names_ = {"output", "r_cache"};
        auto metadata = session_->GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;
        cache_dim_ = std::stoi(metadata.LookupCustomMetadataMap("cache_dim",
                                                                allocator));
        cache_len_ = std::stoi(metadata.LookupCustomMetadataMap("cache_len",
                                                                allocator));
        std::cout << "Kws Model Info:" << std::endl
                  << "\tcache_dim: " << cache_dim_ << std::endl
                  << "\tcache_len: " << cache_len_ << std::endl;

        Reset();


    }

    void KeywordSpotting::Reset() {
        Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        if(mmodel_type == 1){ // ctc model
            cache_.resize(cache_dim_ * cache_len_ * cache_4_, 0.0);
            const int64_t cache_shape[] = {1, cache_dim_, cache_len_, cache_4_};
            cache_ort_ = Ort::Value::CreateTensor<float>(
                    memory_info, cache_.data(), cache_.size(), cache_shape, 4);
            reset_value();
        }else{ // max pooling model
            cache_.resize(cache_dim_ * cache_len_ , 0.0);
            const int64_t cache_shape[] = {1, cache_dim_, cache_len_};
            cache_ort_ = Ort::Value::CreateTensor<float>(
                    memory_info, cache_.data(), cache_.size(), cache_shape, 3);
        }
    }

    void KeywordSpotting::reset_value() {
        if (mdecode_type == DECODE_PREFIX_BEAM_SEARCH) {
            PrefixScore prefix_score;
            prefix_score.s = 1.0;
            prefix_score.ns = 0.0;
            std::vector<int> empty;
            cur_hyps_[empty] = prefix_score;
        } else if (mdecode_type == DECODE_GREEDY_SEARCH) {
            gd_cur_hyps.clear();
        }
    }

    void KeywordSpotting::Forward(
            const std::vector<std::vector<float>> &feats,
            std::vector<std::vector<float>> *prob) {
        prob->clear();
        if (feats.size() == 0) return;
        Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 1. Prepare input
        int num_frames = feats.size();
        int feature_dim = feats[0].size();
        std::vector<float> slice_feats;
        for (int i = 0; i < feats.size(); i++) {
            slice_feats.insert(slice_feats.end(), feats[i].begin(), feats[i].end());
        }
        const int64_t feats_shape[3] = {1, num_frames, feature_dim};
        Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
                memory_info, slice_feats.data(), slice_feats.size(), feats_shape, 3);
        // 2. Ort forward
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(feats_ort));
        inputs.emplace_back(std::move(cache_ort_));
        // ort_outputs.size() == 2
        std::vector<Ort::Value> ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, in_names_.data(), inputs.data(),
                inputs.size(), out_names_.data(), out_names_.size());

        // 3. Update cache
        cache_ort_ = std::move(ort_outputs[1]);

        // 4. Get keyword prob
        float *data = ort_outputs[0].GetTensorMutableData<float>();
        auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
        int num_outputs = type_info.GetShape()[1];
        int output_dim = type_info.GetShape()[2];
        prob->resize(num_outputs);
        for (int i = 0; i < num_outputs; i++) {
            (*prob)[i].resize(output_dim);
            memcpy((*prob)[i].data(), data + i * output_dim,
                   sizeof(float) * output_dim);
        }
    }

    void KeywordSpotting::readToken(const std::string &tokenFile) {
        std::ifstream fin(tokenFile);

        if (fin.is_open()) {
            std::string line;
            while (std::getline(fin, line)) {
                if(mmodel_type==1){
                    std::string token;
                    int value;
                    std::istringstream iss(line);
                    if (iss >> token >> value) {
                        mvocab[token] = value - 1;
                    }
                }else{
                    mmaxpooling_keywords.push_back(line);
                }
            }
            fin.close();
        } else {
            std::cerr << "Error: Unable to open the token file." << std::endl;
        }

    }

    void KeywordSpotting::setKeyWord(const std::string &keyWord) {
        /*keyWord : key word to wakeup.
         * */
        mkey_word = keyWord;
        mkeyword_set.insert(0);  // insert 0 for blank token of ctc.
        for (int idx = 0; idx < keyWord.size(); idx += 3) { // 3byte for chinese char with utf8.
            std::string token = keyWord.substr(idx, 3);
            int toekn_idx = mvocab.at(token);
            if (mvocab.count(token) > 0) {
                if (mkeyword_set.count(toekn_idx) == 0) {
                    mkeyword_set.insert(toekn_idx);
                }
                mkeyword_token.push_back(toekn_idx);
            } else {
                std::cerr << "Can not find" << " " << keyWord << " " << "in vocab. Please check.";
            }
        }
    }

    bool KeywordSpotting::isKeyword(int index) {
        return mkeyword_set.count(index) > 0;
    }

    void KeywordSpotting::UpdateHypotheses(const std::vector<std::pair<std::vector<int>, PrefixScore>> &hpys) {
        cur_hyps_.clear();
        for (auto &item: hpys) {
            // std::vector<int> prefix =  item.first;
            if (item.first.empty()) {
                PrefixScore prefix_score;
                prefix_score.s = 1.0;
                prefix_score.ns = 0.0;
                std::vector<int> empty;
                cur_hyps_[empty] = prefix_score;
            } else {
                // filter illegal prefix case.
                if(item.first.size() > mkeyword_token.size()) {
                    continue;
                }
                cur_hyps_[item.first] = item.second;
            }
        }
        // assert cur_hyps_ is not empty()
        if (!cur_hyps_.empty()){
            cur_hyps_[std::vector<int>()] = PrefixScore{1.0, 0.0};
        }

    }

    void KeywordSpotting::decode_keywords(int stepT, std::vector<std::vector<float>> &probs) {
        /*decode keyword.
         */
        if (mdecode_type == DECODE_GREEDY_SEARCH) {
            //std::cout << "DECODE_GREEDY_SEARCH" << std::endl;
            decode_with_greedy_search(stepT, probs);

        } else if (mdecode_type == DECODE_PREFIX_BEAM_SEARCH) {
            // std::cout << "DECODE_PREFIX_BEAM_SEARCH" << std::endl;
            for (const auto &prob: probs) {
                mGTimeStep += 1;
                decode_ctc_prefix_beam_search(mGTimeStep, prob);
            }
        } else {
            std::cerr << "Not implement yet now.";
        }
    }

    void KeywordSpotting::decode_with_greedy_search(int offset, std::vector<std::vector<float>> &probs) {

        // find index with max-prob in each time step.
        for (int i = 0; i < probs.size(); i++) {
            std::cout << "frame " << std::setw(3) << offset + i;
            auto maxElement = std::max_element(probs[i].begin(), probs[i].end());
            int maxIndex = std::distance(probs[i].begin(), maxElement);
            std::cout << " maxIndex " << std::setw(4) << maxIndex << " prob " << probs[i][maxIndex];
            Token token = {offset + i, maxIndex, probs[i][maxIndex]};
            alignments.emplace_back(token);
            std::cout << std::endl;
        }

        // find hypotheses with token index. just consider one path.
        // it's not update prob when meeting same token, now.
        std::unordered_set<int> seenIds;
        for (const auto &token: alignments) {
            if (token.id != 0 && isKeyword(token.id)) {
                if (seenIds.count(token.id) == 0) {
                    // not see token in current hyp.
                    gd_cur_hyps.push_back(token);
                    seenIds.insert(token.id);
                } //update prob
            } else {
                seenIds.clear();
            }
        }
        alignments.clear();
    }

    void KeywordSpotting::decode_ctc_prefix_beam_search(int stepT, const std::vector<float> &probv) {
        /* Decoding ctc sequence with prefix beam search.
         * ref: https://distill.pub/2017/ctc/
         * python implement
         *      https://modelscope.cn/studios/thuduj12/KWS_Nihao_Xiaojing/file/view/master/stream_kws_ctc.py
         * https://robin1001.github.io/2020/12/11/ctc-search
         * */

//        std::cout << "stepT=" << std::setw(3) << stepT << std::endl;
        if (probv.size() == 0) return;
        std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;

        // 1. First beam prune, only select topk candidates
        std::vector<float> topk_probs;
        std::vector<int> topk_index;
        TopK(probv, opts_.first_beam_size, &topk_probs, &topk_index);

        // filter prob score that is too small.
        std::vector<float> filter_probs;
        std::vector<int> filter_index;
        for (int i = 0; i < opts_.first_beam_size; i++) {
            int idx = topk_index[i];
            float prob = probv[idx];

            if (!mkeyword_set.empty()) {
                if (prob > 0.05 && isKeyword(idx)) {
                    filter_probs.push_back(prob);
                    filter_index.push_back(idx);
                }
            } else {
                if (prob > 0.05) {
                    filter_probs.push_back(prob);
                    filter_index.push_back(idx);
                }
            }
        }

        // handle prefix beam search
        if (!filter_index.empty()) {
            for (int i = 0; i < filter_index.size(); i++) {
                int tokenId = filter_index[i]; // token index of vocab
                float ps = probv[tokenId]; // prob of token
                std::cout << "stepT=" << std::setw(3) << stepT << " tokenid=" << std::setw(4) << tokenId \
                << " proposed i=" << i << " prob=" << std::setprecision(3)  << ps  << std::endl;
                for (const auto &it: cur_hyps_) {//Fixing bug that can't be wakeup in stream-mode"
                    const std::vector<int> &prefix = it.first;
                    const PrefixScore &prefix_score = it.second;
                    print_vector(prefix);
                    if (tokenId == opts_.blank) {
                        // handle ending with blank token. eg 你好问 + ε ->你好问
                        PrefixScore &next_score = next_hyps[prefix];
                        next_score.s = next_score.s + prefix_score.s * ps + prefix_score.ns * ps;
                        next_score.nodes = prefix_score.nodes; // keep the nodes

                    } else if (!prefix.empty() && tokenId == prefix.back()) {
                        if (!(std::abs(prefix_score.ns - 0.0) <= 1e-6)) {
                            // 处理: 你好-好->你好 . 消除alignment中两个blank之间的重复token.
                            PrefixScore &next_score1 = next_hyps[prefix];
                            // update prob of same token.
                            std::vector<Token> next_nodes(prefix_score.nodes); // copy current nodes
                            if (!next_nodes.empty() && ps > next_nodes.back().prob) {
                                next_nodes.back().prob = ps;
                                next_nodes.back().timeStep = stepT;
                            }
                            next_score1.ns = next_score1.ns + prefix_score.ns * ps;
                            next_score1.nodes = next_nodes;
                        }
                        if (!(std::abs(prefix_score.s - 0.0) <= 1e-6)) {
                            // 处理: 你好-好->你好好 . 保留输出序列中的重复字符.
                            std::vector<int> next_prefix(prefix);
                            next_prefix.push_back(tokenId);
                            PrefixScore &next_score2 = next_hyps[next_prefix];
                            next_score2.ns = next_score2.ns + prefix_score.s * ps;
                            Token curToken = {stepT, tokenId, ps};
                            // update nodes from current nodes
                            std::vector<Token> next_nodes(prefix_score.nodes); // copy current nodes
                            next_nodes.push_back(curToken);
                            next_score2.nodes = next_nodes;
                        }

                    } else {
                        //std::cout << "##Not see Token" << std::endl;

                        std::vector<int> next_prefix(prefix);
                        next_prefix.push_back(tokenId);
                        PrefixScore &next_score3 = next_hyps[next_prefix];

                        if (!next_score3.nodes.empty()) {
                            // update prob of same token
                            if (ps > next_score3.nodes.back().prob) {
                                next_score3.nodes.pop_back();
                                Token curToken = {stepT, tokenId, ps};
                                next_score3.nodes.push_back(curToken);
                                next_score3.ns = prefix_score.ns;
                                next_score3.s = prefix_score.s;
                            }
                        } else {
                            std::vector<Token> next_nodes(prefix_score.nodes);      // copy current nodes
                            Token curToken = {stepT, tokenId, ps};
                            next_nodes.push_back(curToken);
                            next_score3.nodes = next_nodes;
                            next_score3.ns = next_score3.ns + prefix_score.s * ps + prefix_score.ns * ps;
                        }
                    }
                }
            }

            // 3 second beam prune. keep topK
            std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                                      next_hyps.end());
            int second_beam_size =
                    std::min(static_cast<int>(arr.size()), opts_.second_beam_size);

            std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                             PrefixScoreCompare);
            arr.resize(second_beam_size);
            std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

            // update
            UpdateHypotheses(arr);
        }

    }

    bool KeywordSpotting::execute_detection(float hitScoreThr) {
        /*　对当前输出的prfix串和关键词进行对比，判断是否唤醒.
         * */
        int start, end, flag = 0;
        float hit_score = 1.0;

        if (mdecode_type == wekws::DECODE_PREFIX_BEAM_SEARCH) {
            for (const auto &it: cur_hyps_) {
                const std::vector<int> &prefix = it.first;
                const std::vector<Token> &nodes = it.second.nodes;
                if (!prefix.empty() && prefix.size() == mkeyword_token.size()) {
                    for (auto i = 0; i < prefix.size(); i++) {
                        flag = (prefix[i] != mkeyword_token[i]) ? 0 : 1;
                        hit_score *= nodes[i].prob;
                        if (i == 0) start = nodes[i].timeStep;
                        if (i == nodes.size() - 1) end = nodes[i].timeStep;
                    }
                }
                if (flag == 1) {
                    cur_hyps_.clear();
                    reset_value();
                    break;
                }
            }
        } else {
            //  std::cout << "cur_hyps size: " << cur_hyps.size() << " kws size: " << this->mkws_ids.size() <<std::endl;
            if (!gd_cur_hyps.empty() && mkeyword_token.size() == gd_cur_hyps.size()) {
                for (auto i = 0; i < gd_cur_hyps.size(); i++) {
                    flag = (gd_cur_hyps[i].id != mkeyword_token[i]) ? 0 : 1;
                    hit_score *= gd_cur_hyps[i].prob;
                    if (i == 0) start = gd_cur_hyps[i].timeStep;
                    if (i == gd_cur_hyps.size() - 1) end = gd_cur_hyps[i].timeStep;
                }
            }
        }
        // find keyword in predicted sequence.
        if (flag == 1 && hit_score >hitScoreThr) {
            std::cout << "hitword=" << mkey_word << std::endl;
            std::cout << "hitscore=" << hit_score << std::endl;
            std::cout << "hitScoreThr=" << hitScoreThr << std::endl;
            std::cout << "start frame=" << start << " end frame=" << end << std::endl;

        }
        return flag;
    }

    void KeywordSpotting::stepClear(){
        mGTimeStep = 0;
    }
}  // namespace wekws
