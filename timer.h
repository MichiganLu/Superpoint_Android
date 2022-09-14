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
#pragma once

#include <float.h>

#include <chrono>
#include <map>
#include <numeric>

#include "numeric_types.h"

inline float CovertMicroTime2Sec(TimeStamp dt) { return dt * 1e-9f; }

// t1 - t2
inline float CovertMicroTimeElapsed2Sec(TimeStamp t1, TimeStamp t2) {
  return t1 > t2 ? (t1 - t2) * 1e-9f : -((t2 - t1) * 1e-9f);
}

// (year - 1) * 372 + ((month - 1) * 31 + day)
inline uint64_t HashTimeData(int year, int month, int day) {
  return (uint64_t)((year - 1) * 372 + ((month - 1) * 31 + day));
}

inline void GetTimeData(int& year, int& month, int& day, int& hour, int& minut,
                        int& sec) {
  time_t rawtime;
  struct tm* timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  year = timeinfo->tm_year + 1900;
  month = timeinfo->tm_mon + 1;
  day = timeinfo->tm_mday;
  hour = timeinfo->tm_hour;
  minut = timeinfo->tm_min;
  sec = timeinfo->tm_sec;
}

/**
 * @brief tictoc timer for multiple entries with counter
 */
class MultiEntryTimer {
  struct TimeData {
    size_t use_cnt_ = 0;
    std::chrono::high_resolution_clock::time_point time_start_;
    float time_cost_ = 0.0f;
    float time_min_ = DBL_MAX;
    float time_max_ = 0.0f;
    float time_sum_ = 0.0f;
  };

 public:
  /**
   * @brief Start timer with entry name
   * @param item_name 		entry name
   */
  void Start(const std::string& item_name);

  /**
   * @brief Stop timer with entry name
   * @param item_name 		entry name
   */
  void Stop(const std::string& item_name);

  /**
   * @brief Add counter of entry name
   * @param item_name 		entry name
   */
  void Count(const std::string& item_name);

  /**
   * @brief Add the counter with 1 for all entries
   */
  void Count();

  /**
   * @brief Stop the timer and add the counter with 1
   * @param item_name 		entry name
   */
  void StopAndCount(const std::string& item_name);

  /**
   * @brief Print elapsed time in milliseconds
   */
  void PrintMilliSeconds(const std::string& item_name);

  /**
   * @brief Print elapsed time in seconds
   */
  void PrintSeconds(const std::string& item_name);

  /**
   * @brief Print elapsed time in minutes
   */
  void PrintMinutes(const std::string& item_name);

  /**
   * @brief Print elapsed time in hours
   */
  void PrintHours(const std::string& item_name);

  /**
   * @brief Print elapsed time in milliseconds
   */
  void PrintMilliSeconds();

  /**
   * @brief Print elapsed time in seconds
   */
  void PrintSeconds();

  /**
   * @brief Print elapsed time in minutes
   */
  void PrintMinutes();

  /**
   * @brief Print elapsed time in hours
   */
  void PrintHours();

  /**
   * @brief Print stat information of entry name
   * @param item_name		entry name
   */
  void Print(const std::string& item_name);

  /**
   * @brief Print stat information of all entries
   */
  void Print();

  /**
   * @brief Get timer stat information of a entry (ms)
   * @param item_name 			entry name
   * @param time_min 		minimum consumption time
   * @param time_max      maximum consumption time
   * @param time_mean 	mean consumption time
   */
  void GetMilliSecondsTime(const std::string& item_name, float& time_min,
                           float& time_max, float& time_mean);

  /**
   * @brief Get timer stat information of a entry (s)
   * @param item_name 			entry name
   * @param time_min 		minimum consumption time
   * @param time_max      maximum consumption time
   * @param time_mean 	mean consumption time
   */
  void GetSecondsTime(const std::string& item_name, float& time_min,
                      float& time_max, float& time_mean);

  /**
   * @brief Get timer stat information of a entry (minute)
   * @param item_name 			entry name
   * @param time_min 		minimum consumption time
   * @param time_max      maximum consumption time
   * @param time_mean 	mean consumption time
   */
  void GetMinutesTime(const std::string& item_name, float& time_min,
                      float& time_max, float& time_mean);

  /**
   * @brief Get timer stat information of a entry (hour)
   * @param item_name 			entry name
   * @param time_min 		minimum consumption time
   * @param time_max      maximum consumption time
   * @param time_mean 	mean consumption time
   */
  void GetHoursTime(const std::string& item_name, float& time_min,
                    float& time_max, float& time_mean);

 private:
  std::map<std::string, TimeData> elements_;
  size_t max_length_ = 0;
};

class SingleEntryTimer {
 public:
  /**
   * @brief multi-types constructor
   */
  SingleEntryTimer() = default;
  SingleEntryTimer(const std::string& name) : name_(name) {}
  SingleEntryTimer(const SingleEntryTimer&) = delete;
  SingleEntryTimer(SingleEntryTimer&&) = delete;

  /**
   * @brief start the timer and set start time
   */
  void Start();

  /**
   * @brief Restart the timer and set the start time
   */
  void Restart();

  /**
   * @brief Pause the timer
   */
  void Pause();

  /**
   * @brief Resume the timer and continue
   */
  void Resume();

  /**
   * @brief Reset the timer and boolean variables
   */
  void Reset();

  /**
   * @brief Compute elapsed time in ms
   * @return duration in ms
   */
  double ElapsedMicroSeconds() const;

  /**
   * @brief Compute elapsed time in seconds
   * @return duration in seconds
   */
  inline double ElapsedSeconds() const { return ElapsedMicroSeconds() / 1e6; };

  /**
   * @brief Compute elapsed time in minutes
   * @return duration in minutes
   */
  inline double ElapsedMinutes() const { return ElapsedSeconds() / 60; }

  /**
   * @brief Compute elapsed time in hours
   * @return duration in hours
   */
  inline double ElapsedHours() const { return ElapsedMinutes() / 60; }

  /**
   * @brief Print elapsed time in milliseconds
   */
  void Print() const;

  /**
   * @brief Print elapsed time in milliseconds
   */
  void PrintMilliSeconds() const;

  /**
   * @brief Print elapsed time in seconds
   */
  void PrintSeconds() const;

  /**
   * @brief Print elasped time in minutes
   */
  void PrintMinutes() const;

  /**
   * @brief Print elapsed time in hours
   */
  void PrintHours() const;

 private:
  const std::string name_ = "";
  bool started_ = false;
  bool paused_ = false;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point pause_time_;
};
