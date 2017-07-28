#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

inline __m512i _interleavelo_epi32(__m512i a, __m512i b) {
  auto c = _mm512_permutexvar_epi64(_mm512_set_epi64(7,3,6,2,5,1,4,0), a);
  auto d = _mm512_permutexvar_epi64(_mm512_set_epi64(7,3,6,2,5,1,4,0), b);
  return _mm512_unpacklo_epi32(c, d);
}

inline __m512i _interleavehi_epi32(__m512i a, __m512i b) {
  auto c = _mm512_permutexvar_epi64(_mm512_set_epi64(7,3,6,2,5,1,4,0), a);
  auto d = _mm512_permutexvar_epi64(_mm512_set_epi64(7,3,6,2,5,1,4,0), b);
  return _mm512_unpackhi_epi32(c, d);
}

inline __m512i _compress_mask(__mmask16 k) {
  auto a = _mm512_set_epi32(15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  return _mm512_mask_compress_epi32(_mm512_undefined_epi32(), k, a);
}

inline void _mask_compressstoreu_epi32(uint32_t* dst, __m512i k, __m512i a) {
  _mm512_storeu_si512(dst, _mm512_permutexvar_epi32(k, a));
}

inline __m512i _rank_get(std::uint64_t* dict, __m512i indice) {
  // calculate the offsets
  auto offsets = _mm512_srli_epi32(indice, 5);

  // IACA says faster on Broadwell
  auto storage = reinterpret_cast<const int32_t*>(dict);
#if 1
  auto bits = _mm512_i32gather_epi32(offsets, storage + 0, 8);
  auto rank = _mm512_i32gather_epi32(offsets, storage + 1, 8);
#else
  auto bits = _mm512_load_si512(storage + 0);
  auto rank = _mm512_load_si512(storage + 1);
#endif

  // we store (bits+bits) to save a cycle (we never count the highest bit)
  indice = _mm512_andnot_si512(indice, _mm512_set1_epi32(0x1F));
  bits   = _mm512_srlv_epi32  (bits  , indice);

#ifndef __AVX512VPOPCNTDQ__
  // constants
  const auto lut = _mm512_set_epi32(
    0x04030302, 0x03020201, 0x03020201, 0x2010100,
    0x04030302, 0x03020201, 0x03020201, 0x2010100,
    0x04030302, 0x03020201, 0x03020201, 0x2010100,
    0x04030302, 0x03020201, 0x03020201, 0x2010100
  );

  // uint8_t  popcount
  auto bitsh = _mm512_srli_epi32(bits, 4);
  auto bitsl = _mm512_and_si512(bits , _mm512_set1_epi8(0x0F));
       bitsh = _mm512_and_si512(bitsh, _mm512_set1_epi8(0x0F));
       bitsl = _mm512_shuffle_epi8(lut, bitsl);
       bitsh = _mm512_shuffle_epi8(lut, bitsh);
       bits  = _mm512_add_epi32(bitsl, bitsh);

  // uint32_t popcount
  bits = _mm512_maskz_dbsad_epu8(0x55, bits, _mm512_setzero_si512(), 0);
  //bits = _mm512_lzcnt_epi32(bits);
#else
  bits = _mm512_popcnt_epi32(bits);
#endif

  return _mm512_add_epi32(rank, bits);
}

int loop(uint32_t* stream, uint16_t* mask, uint32_t* out, uint32_t* coder_iter, std::size_t step, std::uint64_t* dict, std::uint32_t C0, uint32_t* queue_out0, uint32_t* queue_out1) {
  __m512i pmask;
  auto o = _mm512_set1_epi32(C0);

  std::uint32_t* queue_iter = stream;
  std::uint32_t* rank_iter  = stream + 0xdeadbeaf;

  std::size_t queue_step = step;
  std::size_t rank_step  = step + 0xdeadbeaf;

  for (int c_ = 0; c_ < 1024; ++c_) {
    IACA_START
    // load queue
    __m512i c_j = _mm512_load_si512(queue_iter + 1 * queue_step);
    __m512i c_i = _mm512_load_si512(queue_iter + 0 * queue_step);
    __m512i c_k = _mm512_load_si512(queue_iter + 2 * queue_step);
    queue_iter += 16;

    // load ranks
    __m512i r1_j = _rank_get(dict, c_j);
    __m512i r1_i = _mm512_load_si512(rank_iter + 0 * rank_step);
    __m512i r1_k = _mm512_load_si512(rank_iter + 1 * rank_step);
    rank_iter += 16;

    // interleave loads and ranks
    // write out only the used ones (lo part)
    std::uint64_t mask_lo = *mask++;
    pmask = _compress_mask(mask_lo);
    _mask_compressstoreu_epi32(out + 0 * step, pmask, _interleavelo_epi32(c_i , c_j ));
    _mask_compressstoreu_epi32(out + 1 * step, pmask, _interleavelo_epi32(c_j , c_k ));
    _mask_compressstoreu_epi32(out + 2 * step, pmask, _interleavelo_epi32(r1_i, r1_j));
    _mask_compressstoreu_epi32(out + 3 * step, pmask, _interleavelo_epi32(r1_j, r1_k));
    out += _mm_popcnt_u64(mask_lo);

    // write out only the used ones (hi part)
    std::uint64_t mask_hi = *mask++;
    pmask = _compress_mask(mask_hi);
    _mask_compressstoreu_epi32(out + 0 * step, pmask, _interleavehi_epi32(c_i , c_j ));
    _mask_compressstoreu_epi32(out + 1 * step, pmask, _interleavehi_epi32(c_j , c_k ));
    _mask_compressstoreu_epi32(out + 2 * step, pmask, _interleavehi_epi32(r1_i, r1_j));
    _mask_compressstoreu_epi32(out + 3 * step, pmask, _interleavehi_epi32(r1_j, r1_k));
    out += _mm_popcnt_u64(mask_hi);

    // calculate some butterflies vars
    auto n_1w_ = _mm512_sub_epi32(c_k, c_j);

    // calculate the number of zeros before each point
    auto r0_i = _mm512_sub_epi32(c_i, r1_i);
    auto r0_j = _mm512_sub_epi32(c_j, r1_j);
    auto r0_k = _mm512_sub_epi32(c_k, r1_k);

    // calculate the rest of the butterfly vars
    auto n__w1 = _mm512_sub_epi32(r1_k, r1_i);
    auto n__w0 = _mm512_sub_epi32(r0_k, r0_i);
    auto n_1w1 = _mm512_sub_epi32(r1_k, r1_j);

    // right out the queue (zero part)
    // @todo store the mask
    __mmask16 context_mask;
    context_mask = _mm512_cmpgt_epi32_mask(r0_j, r0_i);
    context_mask = _mm512_mask_cmpgt_epi32_mask(context_mask, r0_k, r0_j);
    _mm512_mask_compressstoreu_epi32(queue_out0, context_mask, _mm512_add_epi32(r0_j, o));
    queue_out0 += _mm_popcnt_u64(context_mask);

    // right out the queue (ones part)
    // @todo store the mask
    context_mask = _mm512_cmpgt_epi32_mask(r1_j, r1_i);
    context_mask = _mm512_mask_cmpgt_epi32_mask(context_mask, r1_k, r1_j);
    _mm512_mask_compressstoreu_epi32(queue_out1, context_mask,  _mm512_add_epi32(r1_j, o));
    queue_out0 += _mm_popcnt_u64(context_mask);

    // calculate the limits from the butterfly
    __m512i min = _mm512_maskz_sub_epi32(_mm512_cmpgt_epi32_mask(n_1w_, n__w0), n_1w_, n__w0);
    __m512i max = _mm512_min_epi32(n_1w_, n__w1);

    // calculate the digit and the base
    auto digit = _mm512_sub_epi32(n_1w1, min);
    auto base  = _mm512_add_epi32(_mm512_sub_epi32(max, min), _mm512_set1_epi32(1));

    // store the limits for the coder
    _mm512_storeu_si512(coder_iter +  0, digit);
    _mm512_storeu_si512(coder_iter + 16, base );
    coder_iter += 32;
  }
  IACA_END
  return 0;
}

int main() {}