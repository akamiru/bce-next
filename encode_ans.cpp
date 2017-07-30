#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

__m512i _mm512_mulhi_epi32(__m512i a, __m512i b) {
  auto a_lo = _mm512_srli_epi64(a, 32);
  auto b_lo = _mm512_srli_epi64(b, 32);
  auto c_lo = _mm512_mul_epi32(a_lo, b_lo);
  auto c_hi = _mm512_mul_epi32(a   , b   );
  return _mm512_mask_blend_epi32(0x5555, c_lo, _mm512_srli_epi64(c_hi, 32));
}

template<class T>
inline T* _addr(T* base, std::size_t byte_step) {
  return reinterpret_cast<T*>(
    reinterpret_cast<std::uint8_t*>(base) + byte_step
  );
}

std::uint16_t* encode_ans(
  std::uint32_t* coder_iter,  // where the stream starts
  std::uint32_t* coder_last,  // where the stream ends
                              //   note that we encode from the end to the front!
  std::uint16_t* stream_out   // target to write the stream out to
) {
  // constants
  constexpr std::size_t m512i_size = sizeof(__m512i);

  // state
  auto state_vec0 = _mm512_set1_epi32(0x10000);
  auto state_vec1 = _mm512_set1_epi32(0x10000);
  auto state_vec2 = _mm512_set1_epi32(0x10000);
  auto state_scr0  = UINT64_C(0x100000000);
  auto state_scr1  = UINT64_C(0x100000000);
  auto state_scr2  = UINT64_C(0x100000000);

  while (coder_iter < coder_last) {
    start:
    // IACA_START
    coder_last = _addr(coder_last, -2 * m512i_size);
    auto digit = _mm512_loadu_si512(_addr(coder_last, 0 * m512i_size));
    auto base  = _mm512_loadu_si512(_addr(coder_last, 1 * m512i_size));

    // check if there are big bases that need special treatment
    auto limit_mask = _mm512_cmplt_epi32_mask(base, _mm512_set1_epi32(0xFFFF));

    // rescale the state
    auto overflow      = _mm512_mulhi_epi32(state_vec0, base);
    auto overflow_mask = _mm512_mask_testn_epi32_mask(limit_mask, overflow, overflow);
    auto shift_outd    = _mm512_maskz_compress_epi32(overflow_mask, state_vec0);
    auto shift_outw    = _mm512_cvtepi32_epi16(shift_outd);
    state_vec0             = _mm512_mask_srli_epi32(state_vec0, overflow_mask, state_vec0, 16);

    // write out
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(stream_out), shift_outw);
    stream_out += _mm_popcnt_u64(overflow_mask);

    // encode
    state_vec0 = _mm512_mask_mullo_epi32(state_vec0, limit_mask, state_vec0, base );
    state_vec0 = _mm512_mask_add_epi32  (state_vec0, limit_mask, state_vec0, digit);

    // rotate states in order to remove dependencies
    auto tmp   = state_vec0;
    state_vec0 = state_vec1;
    state_vec1 = state_vec2;
    state_vec2 = tmp;

    // test if there where big codings
    // IACA_END
    if (!_mm512_kortestc(limit_mask, limit_mask)) {
      limit_mask = _mm512_knot(limit_mask);

      // pack left the big codings
      digit = _mm512_maskz_compress_epi32(limit_mask, digit);
      base  = _mm512_maskz_compress_epi32(limit_mask, base );

      // encode
      auto n   = _mm_popcnt_u32(limit_mask);
      auto ans = [&stream_out, &state_scr0, &state_scr1, &state_scr2](auto digit, auto base) {
        IACA_START
        // rescale
        std::uint64_t overflow;
        _mulx_u64(state_scr0, static_cast<uint32_t>(base), &overflow);
        *stream_out  = static_cast<uint32_t>(state_scr0);
         stream_out += 2 * (overflow > 0);
        auto state_shifted = state_scr0 >> 32;
        if (overflow > 0) state_scr0 = state_shifted;

        // encode
        state_scr0 *= static_cast<uint32_t>(base );
        state_scr0 += static_cast<uint32_t>(digit);

        // rotate states in order to remove dependencies
        auto tmp   = state_scr0;
        state_scr0 = state_scr1;
        state_scr1 = state_scr2;
        state_scr2 = tmp;
        IACA_END
      };

      // encode xmm[0]
      auto digit_128 = _mm512_castsi512_si128(digit);
      auto base_128  = _mm512_castsi512_si128(base );
      ans(_mm_cvtsi128_si32(digit_128   ), _mm_cvtsi128_si32(base_128   ));
      if (n ==  1) continue;
      ans(_mm_extract_epi32(digit_128, 1), _mm_extract_epi32(base_128, 1));
      if (n ==  2) continue;
      ans(_mm_extract_epi32(digit_128, 2), _mm_extract_epi32(base_128, 2));
      if (n ==  3) continue;
      ans(_mm_extract_epi32(digit_128, 3), _mm_extract_epi32(base_128, 3));
      if (n ==  4) continue;

      // encode xmm[1]
      digit_128 = _mm512_extracti32x4_epi32(digit, 1);
      base_128  = _mm512_extracti32x4_epi32(base , 1);
      ans(_mm_cvtsi128_si32(digit_128   ), _mm_cvtsi128_si32(base_128   ));
      if (n ==  5) continue;
      ans(_mm_extract_epi32(digit_128, 1), _mm_extract_epi32(base_128, 1));
      if (n ==  6) continue;
      ans(_mm_extract_epi32(digit_128, 2), _mm_extract_epi32(base_128, 2));
      if (n ==  7) continue;
      ans(_mm_extract_epi32(digit_128, 3), _mm_extract_epi32(base_128, 3));
      if (n ==  8) continue;

      // encode xmm[2]
      digit_128 = _mm512_extracti32x4_epi32(digit, 2);
      base_128  = _mm512_extracti32x4_epi32(base , 2);
      ans(_mm_cvtsi128_si32(digit_128   ), _mm_cvtsi128_si32(base_128   ));
      if (n ==  9) continue;
      ans(_mm_extract_epi32(digit_128, 1), _mm_extract_epi32(base_128, 1));
      if (n == 10) continue;
      ans(_mm_extract_epi32(digit_128, 2), _mm_extract_epi32(base_128, 2));
      if (n == 11) continue;
      ans(_mm_extract_epi32(digit_128, 3), _mm_extract_epi32(base_128, 3));
      if (n == 12) continue;

      // encode xmm[3]
      digit_128 = _mm512_extracti32x4_epi32(digit, 3);
      base_128  = _mm512_extracti32x4_epi32(base , 3);
      ans(_mm_cvtsi128_si32(digit_128   ), _mm_cvtsi128_si32(base_128   ));
      if (n == 13) continue;
      ans(_mm_extract_epi32(digit_128, 1), _mm_extract_epi32(base_128, 1));
      if (n == 14) continue;
      ans(_mm_extract_epi32(digit_128, 2), _mm_extract_epi32(base_128, 2));
      if (n == 15) continue;
      ans(_mm_extract_epi32(digit_128, 3), _mm_extract_epi32(base_128, 3));
    }
  }

  // store vector states
  _mm512_storeu_si512(_addr(stream_out, 0 * m512i_size), state_vec0);
  _mm512_storeu_si512(_addr(stream_out, 1 * m512i_size), state_vec1);
  _mm512_storeu_si512(_addr(stream_out, 2 * m512i_size), state_vec2);
  stream_out = _addr(stream_out, 3 * m512i_size);

  // store scalar states
  reinterpret_cast<std::uint64_t*>(stream_out)[0] = state_scr0;
  reinterpret_cast<std::uint64_t*>(stream_out)[1] = state_scr1;
  reinterpret_cast<std::uint64_t*>(stream_out)[2] = state_scr2;

  // return pointer to the end
  return _addr(stream_out, 3 * sizeof(std::uint64_t));
}

int main() {}