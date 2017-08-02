#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

#if 0
__m512i _mm512_mulhi_epi32(__m512i a, __m512i b) {
  auto a_lo = _mm512_srli_epi64(a, 32);
  auto b_lo = _mm512_srli_epi64(b, 32);
  auto c_lo = _mm512_mul_epi32(a_lo, b_lo);
  auto c_hi = _mm512_mul_epi32(a   , b   );
  return _mm512_mask_blend_epi32(0x5555, c_lo, _mm512_srli_epi64(c_hi, 32));
}
#endif

// constants
constexpr std::size_t m512i_size = sizeof(__m512i);
constexpr std::size_t small_bits = 23;
constexpr std::size_t small_shft =  8;
constexpr std::size_t large_bits = 52;
constexpr std::size_t large_shft = 32;
constexpr std::size_t small_smin = 1 << (small_bits - small_shft);
constexpr std::size_t large_smin = 1 << (large_bits - large_shft);

template<class T>
inline T* _addr(T* base, std::size_t byte_step) {
  return reinterpret_cast<T*>(
    reinterpret_cast<std::uint8_t*>(base) + byte_step
  );
}

std::uint8_t* encode_ans(
  std::uint32_t* coder_iter,  // where the stream starts
  std::uint32_t* coder_last,  // where the stream ends
                              //   note that we encode from the end to the front!
  std::uint8_t*  stream_out   // target to write the stream out to
) {
  // small states, 3 for pipelining
  auto state_sml0 = _mm512_set1_epi32(small_smin);
  auto state_sml1 = _mm512_set1_epi32(small_smin);
  auto state_sml2 = _mm512_set1_epi32(small_smin);
  // large states, 3 for pipelining (might not be worth it)
  //auto state_lrg0 = _mm512_set1_epi64(large_smin);
  //auto state_lrg1 = _mm512_set1_epi64(large_smin);
  //auto state_lrg2 = _mm512_set1_epi64(large_smin);

  // we write from a high adress to a low so we need to
  __m128i stream_buf;
  __m128i shuffle_mk[] = {
    // shuffle masks that move each element left by 1 and clear the low bytes
    _mm_set_epi8(0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00),
    _mm_set_epi8(0x0E,0x0D,0x0C,0x0B,0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF),
    _mm_set_epi8(0x0D,0x0C,0x0B,0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF),
    _mm_set_epi8(0x0C,0x0B,0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x0B,0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x0A,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x07,0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x06,0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x05,0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x04,0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x03,0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x02,0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x01,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF),
    _mm_set_epi8(0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF)
  };

  while (coder_iter < coder_last) {
    IACA_START
    // load data
    coder_last = _addr(coder_last, -2 * m512i_size);
    auto digit = _mm512_loadu_si512(_addr(coder_last, 0 * m512i_size));
    auto base  = _mm512_loadu_si512(_addr(coder_last, 1 * m512i_size));

    // check if there are small codings
    auto small_mask = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(32), base);

    // check for overflow
    auto oflow_test = _mm512_mullo_epi32(state_sml0, base);
    auto oflow_mask = _mm512_mask_cmpgt_epi32_mask(small_mask, oflow_test, _mm512_set1_epi32((1 << small_bits) - 1));

    // shift out
    auto state_shft = _mm512_cvtepi32_epi8(_mm512_maskz_compress_epi32(oflow_mask, state_sml0));
    auto oflow_n    = _mm_popcnt_u64(oflow_mask);
    stream_buf      = _mm_shuffle_epi8(stream_buf, _mm_load_si128(shuffle_mk + oflow_n));
    stream_buf      = _mm_or_si128(stream_buf, state_shft);  // @todo if vpermi2b has a low latency then this might be improved
    stream_out     -= oflow_n;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(stream_out), stream_buf);  // partially overrides

    // rescale
    state_sml0 = _mm512_mask_srli_epi32(state_sml0, oflow_mask, state_sml0, small_shft);

    // actual encoding
    state_sml0 = _mm512_mask_mullo_epi32(state_sml0, small_mask, state_sml0, base );
    state_sml0 = _mm512_mask_add_epi32  (state_sml0, small_mask, state_sml0, digit);

    // rotate the states to reduce dependency
    auto tmp   = state_sml0;
    state_sml0 = state_sml1;
    state_sml1 = state_sml2;
    state_sml2 = tmp;

    if (!_mm512_kortestc(small_mask, small_mask)) {
      // @todo Add fallback for large bases
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(stream_out), tmp);  // just to keep the branch
    }
  }
  IACA_END

  // store small states
  _mm512_storeu_si512(_addr(stream_out, 0 * m512i_size), state_sml0);
  _mm512_storeu_si512(_addr(stream_out, 1 * m512i_size), state_sml1);
  _mm512_storeu_si512(_addr(stream_out, 2 * m512i_size), state_sml2);
  stream_out = _addr(stream_out, 3 * m512i_size);

  // store large states
  //_mm512_storeu_si512(_addr(stream_out, 0 * m512i_size), state_lrg0);
  //_mm512_storeu_si512(_addr(stream_out, 1 * m512i_size), state_lrg1);
  //_mm512_storeu_si512(_addr(stream_out, 2 * m512i_size), state_lrg2);
  //stream_out = _addr(stream_out, 3 * m512i_size);

  // return a pointer to the end
  return stream_out;
}

int main() {}
