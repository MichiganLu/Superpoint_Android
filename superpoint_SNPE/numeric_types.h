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
// @author: Yilun Zhang
// @date:   2021-01-07
//

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <iostream>

/*==================== ids ==========================*/
using TimeStamp = uint64_t;
using FrameId = uint64_t;
using SensorID = int32_t;
using PointID = uint32_t;
using FeatureID = uint32_t;
using ClusterID = uint32_t;
// using Index = uint32_t;

/// will be faster using constexpr
constexpr uint32_t kInvaliduint32 = std::numeric_limits<uint32_t>::max();
constexpr uint64_t kInvaliduint64 = std::numeric_limits<uint64_t>::max();
constexpr int64_t Invalidint64 = std::numeric_limits<int64_t>::max();
constexpr TimeStamp kInvalidTimeStamp = std::numeric_limits<TimeStamp>::max();
constexpr FrameId kInvalidFrameId = std::numeric_limits<FrameId>::max();
constexpr SensorID kInvalidSensorId = std::numeric_limits<SensorID>::max();
constexpr PointID kInvalidPointId = std::numeric_limits<PointID>::max();
constexpr FeatureID kInvalidFeatureId = std::numeric_limits<FeatureID>::max();
constexpr ClusterID kInvalidClusterId = std::numeric_limits<ClusterID>::max();
// constexpr Index kInvalidIndex = std::numeric_limits<Index>::max();

using VecTimeStamps = std::vector<TimeStamp>;
using VecFrameIds = std::vector<FrameId>;
using VecSensorIds = std::vector<SensorID>;
using VecPointIds = std::vector<PointID>;
using VecFeatureIds = std::vector<FeatureID>;
using VecClusterIds = std::vector<ClusterID>;
// using VecIndexs = std::vector<Index>;
using VecUChars = std::vector<unsigned char>;
using VecInts = std::vector<int>;
using VecFloats = std::vector<float>;
using VecDoubles = std::vector<double>;

// number type
using NumericType = double;

/*==================== traits =======================*/
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/* *************************************************************************
 * References:
 * 1.
 * https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 * 2. https://floating-point-gui.de/errors/comparison/
 * *************************************************************************
 */
template <typename Scalar,
          typename = enable_if_t<std::is_floating_point<Scalar>::value>>
inline bool fpEqual(Scalar a, Scalar b, Scalar tol,
                    bool check_relative_also = false) {
  using std::abs;
  using std::isinf;
  using std::isnan;

  Scalar DOUBLE_MIN_NORMAL = std::numeric_limits<Scalar>::min() + 1.0;
  Scalar larger = (abs(b) > abs(a)) ? abs(b) : abs(a);

  // handle NaNs
  if (isnan(a) || isnan(b)) {
    return isnan(a) && isnan(b);
  }
  // handle inf
  else if (isinf(a) || isinf(b)) {
    return isinf(a) && isinf(b);
  }
  // If the two values are zero or both are extremely close to it
  // relative error is less meaningful here
  else if (a == 0 || b == 0 || (abs(a) + abs(b)) < DOUBLE_MIN_NORMAL) {
    return abs(a - b) <= tol * DOUBLE_MIN_NORMAL;
  }
  // Check if the numbers are really close.
  // Needed when comparing numbers near zero or tol is in vicinity.
  else if (abs(a - b) <= tol) {
    return true;
  }
  // Check for relative error
  else if (abs(a - b) <=
               tol * std::min(larger, std::numeric_limits<Scalar>::max()) &&
           check_relative_also) {
    return true;
  }

  return false;
}

/* *************************************************************************
 * Check Valid funciton for all types.
 *
 */

template <typename Scalar, typename Enable = void>
struct Check {
  static bool IsValid(Scalar value) {
    std::cout << "Un defined Check.." << std::endl;
    return true;
  }
};

template <typename Scalar>
struct Check<Scalar, enable_if_t<std::is_floating_point<Scalar>::value>> {
  static bool IsValid(Scalar value) { return !std::isnan(value); }
};

template <typename Scalar>
struct Check<Scalar, enable_if_t<std::is_unsigned<Scalar>::value>> {
  static bool IsValid(Scalar value) {
    return value != std::numeric_limits<Scalar>::max();
  }
};

template <typename Scalar>
struct Check<Scalar, enable_if_t<std::is_signed<Scalar>::value &&
                                 std::is_integral<Scalar>::value>> {
  static bool IsValid(Scalar value) { return value > 0; }
};

// template<typename Scalar, typename =
// enable_if_t<std::is_unsigned<Scalar>::value>> struct CheckValid{
//     static bool Check(Scalar value) { return value !=
//     std::numeric_limits<Scalar>::max(); }
// };

