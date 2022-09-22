// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// @author: Yilun Zhang, Yi Liu, Kanzhi Wu
// @date:   2021-01-07
//
#include "timer.h"

#include <assert.h>
#include <glog/logging.h>


void MultiEntryTimer::Start(const std::string& item_name) {
  if (item_name.length() > max_length_) max_length_ = item_name.length();

  auto iter = elements_.find(item_name);
  if (iter == elements_.end()) {
    elements_.insert(
        std::map<std::string, TimeData>::value_type(item_name, TimeData()));
    iter = elements_.find(item_name);
  }

  assert(iter != elements_.end());
  iter->second.time_start_ = std::chrono::high_resolution_clock::now();
}

void MultiEntryTimer::Stop(const std::string& item_name) {
  auto iter = elements_.find(item_name);
  if (iter == elements_.end()) return;

  iter->second.time_cost_ +=
      std::chrono::duration_cast<
          std::chrono::duration<float, std::ratio<1, 1000> > >(
          std::chrono::high_resolution_clock::now() - iter->second.time_start_)
          .count();
}

void MultiEntryTimer::Count(const std::string& item_name) {
  auto iter = elements_.find(item_name);
  if (iter == elements_.end()) return;

  iter->second.use_cnt_++;
  if (iter->second.time_cost_ < iter->second.time_min_)
    iter->second.time_min_ = iter->second.time_cost_;
  if (iter->second.time_cost_ > iter->second.time_max_)
    iter->second.time_max_ = iter->second.time_cost_;
  iter->second.time_sum_ += iter->second.time_cost_;
  iter->second.time_cost_ = 0.0f;
}

void MultiEntryTimer::StopAndCount(const std::string& item_name) {
  Stop(item_name);
  Count(item_name);
}

void MultiEntryTimer::Count() {
  for (auto iter = elements_.begin(); iter != elements_.end(); ++iter) {
    iter->second.use_cnt_++;
    if (iter->second.time_cost_ < iter->second.time_min_)
      iter->second.time_min_ = iter->second.time_cost_;
    if (iter->second.time_cost_ > iter->second.time_max_)
      iter->second.time_max_ = iter->second.time_cost_;
    iter->second.time_sum_ += iter->second.time_cost_;
    iter->second.time_cost_ = 0.0f;
  }
}

#define TIMER_PRINT_INTER(unit, multi)                                       \
  {                                                                          \
    std::string name = iter->first;                                          \
    if (name.length() < max_length_ + 3) {                                   \
      name.append(max_length_ + 3 - name.length(), ' ');                     \
    }                                                                        \
    std::string time_min = std::to_string(iter->second.time_min_ / multi);   \
    std::string time_max = std::to_string(iter->second.time_max_ / multi);   \
    std::string time_avg = std::to_string((iter->second.time_sum_ / multi) / \
                                          iter->second.use_cnt_);            \
    std::string time_sum = std::to_string(iter->second.time_sum_ / multi);   \
    time_min.resize(5);                                                      \
    time_max.resize(5);                                                      \
    time_avg.resize(5);                                                      \
    time_sum.resize(7);                                                      \
    char buffer[100];                                                        \
    sprintf(buffer, "%s -> min:%s, max:%s, avg:%s, sum:%s %s\n",            \
            name.c_str(), time_min.c_str(), time_max.c_str(),                \
            time_avg.c_str(), time_sum.c_str(), #unit);                      \
    LOG(INFO) << buffer;                                                     \
  }

#define GEN_TIMER_PRINT(func_name, unit, multi)                            \
  void MultiEntryTimer::func_name(const std::string& item_name) {          \
    auto iter = elements_.find(item_name);                                 \
    if (iter == elements_.end()) return;                                   \
    if (iter->second.use_cnt_ == 0) return;                                \
    TIMER_PRINT_INTER(unit, multi)                                         \
  }                                                                        \
  void MultiEntryTimer::func_name() {                                      \
    for (auto iter = elements_.begin(); iter != elements_.end(); ++iter) { \
      if (iter->second.use_cnt_ == 0) continue;                            \
      TIMER_PRINT_INTER(unit, multi)                                       \
    }                                                                      \
    LOG(INFO) << "\n";                                                     \
  }

#define GEN_TIMER_GET(func_name, multi)                                  \
  void MultiEntryTimer::func_name(const std::string& item_name,          \
                                  float& time_min, float& time_max,      \
                                  float& time_avg) {                     \
    auto iter = elements_.find(item_name);                               \
    if (iter == elements_.end()) return;                                 \
    if (iter->second.use_cnt_ == 0) return;                              \
    time_min = iter->second.time_min_ / multi;                           \
    time_max = iter->second.time_max_ / multi;                           \
    time_avg = (iter->second.time_sum_ / multi) / iter->second.use_cnt_; \
  }

GEN_TIMER_PRINT(PrintMilliSeconds, ms, 1.0)
GEN_TIMER_PRINT(PrintSeconds, s, 1000.0)
GEN_TIMER_PRINT(PrintMinutes, minute, 60000.0)
GEN_TIMER_PRINT(PrintHours, h, 3600000.0)

void MultiEntryTimer::Print(const std::string& item_name) {
  PrintMilliSeconds(item_name);
}

void MultiEntryTimer::Print() { PrintMilliSeconds(); }

GEN_TIMER_GET(GetMilliSecondsTime, 1.0)
GEN_TIMER_GET(GetSecondsTime, 1000.0)
GEN_TIMER_GET(GetMinutesTime, 60000.0)
GEN_TIMER_GET(GetHoursTime, 3600000.0)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SingleEntryTimer::Start() {
  started_ = true;
  paused_ = false;
  start_time_ = std::chrono::high_resolution_clock::now();
}

void SingleEntryTimer::Restart() {
  started_ = false;
  Start();
}

void SingleEntryTimer::Pause() {
  paused_ = true;
  pause_time_ = std::chrono::high_resolution_clock::now();
}

void SingleEntryTimer::Resume() {
  paused_ = false;
  start_time_ += std::chrono::high_resolution_clock::now() - pause_time_;
}

void SingleEntryTimer::Reset() {
  started_ = false;
  paused_ = false;
}

double SingleEntryTimer::ElapsedMicroSeconds() const {
  if (!started_) return 0.0;
  if (paused_)
    return std::chrono::duration_cast<std::chrono::microseconds>(pause_time_ -
                                                                 start_time_)
        .count();
  else
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start_time_)
        .count();
}

/** @todo using crispy logging implementation instead */
void SingleEntryTimer::Print() const {}

void SingleEntryTimer::PrintMilliSeconds() const {
  char buffer[100];
  sprintf(buffer, "Elapsed time: %.3f [ms]\n", ElapsedMicroSeconds() / 1000.0);
  LOG(INFO) << buffer;
}

void SingleEntryTimer::PrintSeconds() const {
  char buffer[100];
  sprintf(buffer, "Elapsed time: %.3f [seconds]\n", ElapsedSeconds());
  LOG(INFO) << buffer;
}

void SingleEntryTimer::PrintMinutes() const {
  char buffer[100];
  sprintf(buffer, "Elapsed time: %.3f [minutes]\n", ElapsedMinutes());
  LOG(INFO) << buffer;
}

void SingleEntryTimer::PrintHours() const {
  char buffer[100];
  sprintf(buffer, "Elapsed time: %.3f [hours]\n", ElapsedHours());
  LOG(INFO) << buffer;
}

