#include <cstdint>
#include <cmath>
#include "immintrin.h"

__m512 div22_by_lt32(__m512 a, __m512i b) {
  // reciprocals: ceiled to next float
  const __m512 t1 = _mm512_set_ps(
    0.032258067516, 0.033333333333, 0.034482762621, 0.035714285714, 0.037037037037, 0.038461538462, 0.040000003000, 0.041666666667,
    0.043478260870, 0.045454545455, 0.047619047619, 0.050000000000, 0.052631578947, 0.055555555556, 0.058823529412, 0.062500000000
  );
  const __m512 t2 = _mm512_set_ps(
    0.066666666667, 0.071428571429, 0.076923076923, 0.083333333333, 0.090909090909, 0.100000000000, 0.111111111111, 0.125000000000,
    0.142857142857, 0.166666666667, 0.200000000000, 0.250000000000, 0.333333333333, 0.500000000000, 1.000000000000, 1.000000000000
  );
  // vpermi2ps to destroy b
  auto rcp = _mm512_permutex2var_ps(t2, b, t1);
  // divide
  auto c = _mm512_fmadd_round_ps(a, rcp, _mm512_set1_ps(1<<23), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  return _mm512_sub_ps(c, _mm512_set1_ps(1<<23));  // floor c
}

template<class T>
inline T* _addr(T* base, std::size_t byte_step) {
  return reinterpret_cast<T*>(
    reinterpret_cast<std::uint8_t*>(base) + byte_step
  );
}

// constants
constexpr std::size_t m512i_size = sizeof(__m512i);
constexpr std::size_t small_bits = 23;
constexpr std::size_t small_shft =  8;
constexpr std::size_t large_bits = 52;
constexpr std::size_t large_shft = 32;
constexpr std::size_t small_smin = 1 << (small_bits - small_shft);
constexpr std::size_t large_smin = 1 << (large_bits - large_shft);

__m512i decode_ans(
  __m512&        state_sml0,  // current state, the rolling is done in the calling loop
  __m512i        base_int32,  // base to decode with
  std::uint8_t*& stream_ptr   // where to read bits
) {
  // check if there are small codings
  auto small_mask = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(32), base_int32);

  __m512i digit_int32;
  if (!_mm512_kortestc(small_mask, small_mask)) {
    // @todo Add fallback for large bases
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(stream_ptr), base_int32);  // just to keep the branch
  }

  // convert to float (should this use the mask?)
  auto base_float = _mm512_cvtepi32_ps(base_int32);

  // decode the symbol
  auto state_sml1 = div22_by_lt32(state_sml0, base_int32);
  auto digit      = _mm512_fnmadd_ps(state_sml1, base_float, state_sml0);  // digit = s0 % b = s0 - s1 * b

  // test for underflow NOTE: we could compare to c in div22_by_lt32 in order to skip latency of the sub
  auto uflow_mask = _mm512_cmp_ps_mask(_mm512_set1_ps(small_smin), state_sml1, _CMP_GT_OQ);

  // load bytes
  stream_ptr -= _mm_popcnt_u64(uflow_mask);
  auto stream_in  = _mm_loadu_si128(reinterpret_cast<__m128i*>(stream_ptr));
  auto stream_int = _mm512_maskz_expand_epi32(uflow_mask, _mm512_cvtepi8_epi32(stream_in));
  auto stream_flt = _mm512_cvtepi32_ps(stream_int);
  // rescale and update state
  state_sml0 = _mm512_mask_fmadd_ps(state_sml1, uflow_mask, _mm512_set1_ps(256), stream_flt);

  // convert and return
  return _mm512_mask_cvtps_epi32(digit_int32, small_mask, digit);
}

void minimal_loop(__m512i* base, std::uint8_t*  stream_ptr) {
  auto state_sml0 = _mm512_set1_ps(small_smin);  // actually loaded from stream_ptr
  auto state_sml1 = _mm512_set1_ps(small_smin);  // actually loaded from stream_ptr
  auto state_sml2 = _mm512_set1_ps(small_smin);  // actually loaded from stream_ptr

  for (__m512i* be = base + 1024; base < be; base++) {
    // IACA_START
    *base = decode_ans(state_sml0, *base, stream_ptr);

    auto tmp   = state_sml0;
    state_sml0 = state_sml1;
    state_sml1 = state_sml2;
    state_sml2 = tmp;
  }
  // IACA_END
}
