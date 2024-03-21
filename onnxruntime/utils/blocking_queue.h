// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

#ifndef UTILS_BLOCKING_QUEUE_H_
#define UTILS_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <limits>
#include <mutex>
#include <queue>
#include <utility>

namespace wenet {

#define WENET_DISALLOW_COPY_AND_ASSIGN(Type) \
  Type(const Type&) = delete;                \
  Type& operator=(const Type&) = delete;

template <typename T>
class BlockingQueue {
    /*这段代码定义了一个线程安全的阻塞队列，可以在多线程环境下进行安全的插入和移除操作。同时，它使用条件变量来进行线程间的同步和通信，
     以实现阻塞和唤醒线程的功能。当队列已满时，插入操作会被阻塞，直到队列有空闲位置。当队列为空时，移除操作会被阻塞，直到队列有元素可供移除。
     这种阻塞队列的设计可以用于实现生产者-消费者模型，其中生产者线程向队列中插入数据，而消费者线程从队列中获取数据。当队列为空时，消费者线程会被阻塞，
     直到有数据可供消费。当队列已满时，生产者线程会被阻塞，直到有空闲位置可以插入数据。
     注意，该代码中使用了C++11中的互斥锁（std::mutex）、条件变量（std::condition_variable），以及队列（std::queue）等线程同步和容器类。
     */
 public:
  explicit BlockingQueue(size_t capacity = std::numeric_limits<int>::max())
      : capacity_(capacity) {}

  void Push(const T& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.size() >= capacity_) {
        not_full_condition_.wait(lock);
      }
      queue_.push(value);
    }
    not_empty_condition_.notify_one();
  }

  void Push(T&& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.size() >= capacity_) {
        not_full_condition_.wait(lock);
      }
      queue_.push(std::move(value));
    }
    not_empty_condition_.notify_one();
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty()) {
      not_empty_condition_.wait(lock);
    }
    T t(std::move(queue_.front()));
    queue_.pop();
    not_full_condition_.notify_one();
    return t;
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void Clear() {
    while (!Empty()) {
      Pop();
    }
  }

 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_full_condition_;
  std::condition_variable not_empty_condition_;
  std::queue<T> queue_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace wenet

#endif  // UTILS_BLOCKING_QUEUE_H_
