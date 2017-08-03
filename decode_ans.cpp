#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

__m512i _mm512_mulhi_epi32(__m512i a, __m512i b) {
  auto a_lo = _mm512_srli_epi64(a, 32);
  auto b_lo = _mm512_srli_epi64(b, 32);
  auto c_lo = _mm512_mul_epi32(a_lo, b_lo);
  auto c_hi = _mm512_mul_epi32(a   , b   );
  //return _mm512_mask_blend_epi32(0x5555, c_lo, _mm512_srli_epi64(c_hi, 32));
  return _mm512_mask_alignr_epi32(c_lo, 0x5555, c_hi, c_hi, 1);
}


__m512 div22_by_lt32(__m512 a, __m512i b) {
  // reciprocals: ceiled to next float
  const __m512 t1 = _mm512_set_ps(
    0.032258067280, 0.033333335072, 0.034482762218, 0.035714287311, 0.037037037313, 0.038461539894, 0.040000002831, 0.041666667908,
    0.043478261679, 0.045454546809, 0.047619048506, 0.050000000745, 0.052631579340, 0.055555555969, 0.058823529631, 0.062500000000
  );
  const __m512 t2 = _mm512_set_ps(
    0.066666670144, 0.071428574622, 0.076923079789, 0.083333335817, 0.090909093618, 0.100000001490, 0.111111111939, 0.125000000000,
    0.142857149243, 0.166666671634, 0.200000002980, 0.250000000000, 0.333333343267, 0.500000000000, 1.000000000000, 1.000000000000
  );
  // vpermi2ps to destroy b
  auto rcp = _mm512_permutex2var_ps(t2, b, t1);
  // divide
  auto c = _mm512_fmadd_round_ps(a, rcp, _mm512_set1_ps(1<<23), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  return _mm512_sub_ps(c, _mm512_set1_ps(1<<23));  // floor c
}

__m512i div22_by_lt32(__m512i a, __m512i b) {
  // reciprocals: ceiled to next float
  const __m512i t1 = _mm512_set_epi32(
    0x08421085, 0x08888889, 0x08d3dcb1, 0x0924924a, 0x097b425f, 0x09d89d8a, 0x0a3d70a4, 0x0aaaaaab,
    0x0b21642d, 0x0ba2e8bb, 0x0c30c30d, 0x0ccccccd, 0x0d79435f, 0x0e38e38f, 0x0f0f0f10, 0x10000000
  );
  const __m512i t2 = _mm512_set_epi32(
    0x11111112, 0x12492493, 0x13b13b14, 0x15555556, 0x1745d175, 0x1999999a, 0x1c71c71d, 0x20000000,
    0x24924925, 0x2aaaaaab, 0x33333334, 0x40000000, 0x55555556, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFF
  );
  // vpermi2ps to destroy b
  auto rcp = _mm512_permutex2var_epi32(t2, b, t1);
  // divide
  return _mm512_mulhi_epi32(a, rcp);  // floor c
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

inline __m512i decode_ans(
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
  auto stream_in  = _mm_loadu_si128(reinterpret_cast<__m128i*>(stream_ptr));
  auto stream_int = _mm512_maskz_expand_epi32(uflow_mask, _mm512_cvtepi8_epi32(stream_in));
  auto stream_flt = _mm512_cvtepi32_ps(stream_int);
  stream_ptr += _mm_popcnt_u64(uflow_mask);

  // rescale and update state
  state_sml0 = _mm512_mask_fmadd_ps(state_sml1, uflow_mask, _mm512_set1_ps(256), stream_flt);

  // convert and return
  return _mm512_mask_cvtps_epi32(digit_int32, small_mask, digit);
}

inline __m512i decode_ans_div(
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
  auto state_sml1 = _mm512_roundscale_ps(_mm512_div_ps(state_sml0, base_float), 1);
  auto digit      = _mm512_fnmadd_ps(state_sml1, base_float, state_sml0);  // digit = s0 % b = s0 - s1 * b

  // test for underflow NOTE: we could compare to c in div22_by_lt32 in order to skip latency of the sub
  auto uflow_mask = _mm512_cmp_ps_mask(_mm512_set1_ps(small_smin), state_sml1, _CMP_GT_OQ);

  // load bytes
  auto stream_in  = _mm_loadu_si128(reinterpret_cast<__m128i*>(stream_ptr));
  auto stream_int = _mm512_maskz_expand_epi32(uflow_mask, _mm512_cvtepi8_epi32(stream_in));
  auto stream_flt = _mm512_cvtepi32_ps(stream_int);
  stream_ptr += _mm_popcnt_u64(uflow_mask);

  // rescale and update state
  state_sml0 = _mm512_mask_fmadd_ps(state_sml1, uflow_mask, _mm512_set1_ps(256), stream_flt);

  // convert and return
  return _mm512_mask_cvtps_epi32(digit_int32, small_mask, digit);
}

inline __m512i decode_ans_int(
  __m512i&       state_sml0,  // current state, the rolling is done in the calling loop
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

  // decode the symbol
  auto state_sml1 = div22_by_lt32(state_sml0, base_int32);
  auto digit      = _mm512_sub_epi32(state_sml0, _mm512_mullo_epi32(state_sml1, base_int32));  // digit = s0 % b = s0 - s1 * b

  // test for underflow NOTE: we could compare to c in div22_by_lt32 in order to skip latency of the sub
  auto uflow_mask = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(small_smin), state_sml1);

  // load bytes
  auto stream_in  = _mm_loadu_si128(reinterpret_cast<__m128i*>(stream_ptr));
  auto stream_int = _mm512_maskz_expand_epi32(uflow_mask, _mm512_cvtepi8_epi32(stream_in));
  stream_ptr += _mm_popcnt_u64(uflow_mask);

  // shift left and add
  state_sml0 = _mm512_mask_slli_epi32(state_sml1, uflow_mask, state_sml1, 8);
  state_sml0 = _mm512_add_epi32(state_sml0, stream_int);

  // add and return
  return digit;
}

void minimal_loop(__m512i* base, std::uint8_t*  stream_ptr) {
  auto state_sml0 = _mm512_set1_ps(small_smin);  // actually loaded from stream_ptr
  auto state_sml1 = _mm512_set1_ps(small_smin);  // actually loaded from stream_ptr
  auto state_sml2 = _mm512_set1_ps(small_smin);  // actually loaded from stream_ptr

  for (__m512i* be = base + 1024; base < be; base++) {
    //IACA_START
    *base = decode_ans(state_sml0, *base, stream_ptr);
    //*base = decode_ans_div(state_sml0, *base, stream_ptr);

    // auto tmp   = state_sml0;
    // state_sml0 = state_sml1;
    // state_sml1 = state_sml2;
    // state_sml2 = tmp;
  }
  //IACA_END
}

void minimal_loop_int(__m512i* base, std::uint8_t*  stream_ptr) {
  auto state_sml0 = _mm512_set1_epi32(small_smin);  // actually loaded from stream_ptr
  auto state_sml1 = _mm512_set1_epi32(small_smin);  // actually loaded from stream_ptr
  auto state_sml2 = _mm512_set1_epi32(small_smin);  // actually loaded from stream_ptr

  for (__m512i* be = base + 1024; base < be; base++) {
    // IACA_START
    *base = decode_ans_int(state_sml0, *base, stream_ptr);

    // auto tmp   = state_sml0;
    // state_sml0 = state_sml1;
    // state_sml1 = state_sml2;
    // state_sml2 = tmp;
  }
  // IACA_END
}

int main() {}