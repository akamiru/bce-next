#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

inline void _interleavelo_compress_and_store(uint32_t* dst, __mmask16 k, __m512i a, __m512i b) {
  auto shuffle_mask = _mm512_set_epi32(23, 7,22, 6,21, 5,20, 4,19, 3,18, 2,17, 1,16, 0);
  shuffle_mask = _mm512_maskz_compress_epi32(k, shuffle_mask);
  _mm512_storeu_si512(dst, _mm512_permutex2var_epi32(a, shuffle_mask, b));
}

inline void _interleavehi_compress_and_store(uint32_t* dst, __mmask16 k, __m512i a, __m512i b) {
  auto shuffle_mask = _mm512_set_epi32(31,15,30,14,29,13,28,12,27,11,26,10,25, 9,24, 8);
  shuffle_mask = _mm512_maskz_compress_epi32(k, shuffle_mask);
  _mm512_storeu_si512(dst, _mm512_permutex2var_epi32(a, shuffle_mask, b));
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
  bits = _mm512_maskz_dbsad_epu8(0x5555, bits, _mm512_setzero_si512(), 0);
#else
  bits = _mm512_popcnt_epi32(bits);
#endif

  return _mm512_add_epi32(rank, bits);
}

inline std::uint32_t* _addr(std::uint32_t* base, std::size_t byte_step) {
  return reinterpret_cast<std::uint32_t*>(
    reinterpret_cast<std::uint8_t*>(base) + byte_step
  );
}

int encode_main (
  // The queues
  //  current
  std::uint32_t* queue_iter,  // start of the queue
  std::uint32_t* queue_last,  // end   of the queue
  // next (queue_out0 MAY BE queue_iter)
  std::uint32_t* queue_out0,  // where to write out the queue (zero)
  std::uint32_t* queue_out1,  // where to write out the queue (ones)

  // cached values (should fit L1 or L2)
  std::uint32_t* cache_iter,  // start of the cached values
  std::uint32_t* cache_next,  // where to write the next cached values
  std::size_t    cache_step,  // step between the cache columns
                              // columns: c_i, c_k, r1_i, unused/coder, r1_k
                              //   we dont use column 3 to save registers (addressing doesn't allow *3)
                              //   L1/L2 shouldn't be effected by this
                              //   unused memory overhead is not that big
                              //   if we interleave all 8 caches we can use
                              //   that one big free chunk for the coder ! 

  // the rank dictionary
  std::uint64_t* rdict_base,  // base address of the dictionary
  std::uint32_t  count_zero,  // number of zeros in this dictionary

  // stream of the selection masks
  std::uint16_t* smask_iter,
  std::uint16_t* smask_out0,
  std::uint16_t* smask_out1,

  // where to store the coder information
  std::uint32_t* coder_iter   // should use the free column in the cache
) {
  // constants
  constexpr std::size_t vector_size = sizeof(__m512i);

  // loop invariants
  auto c_z = _mm512_set1_epi32(count_zero);  // broadcast the amountof zeros

  // adjustments
  cache_step *= sizeof(*queue_iter);  // adjust for _addr()

  while (queue_iter < queue_last) {
    IACA_START
    // load queue
    auto c_j   = _mm512_load_si512(queue_iter);
    auto c_i   = _mm512_load_si512(_addr(cache_iter, 0 * cache_step));
    auto c_k   = _mm512_load_si512(_addr(cache_iter, 1 * cache_step));
    queue_iter = _addr(queue_iter, vector_size);

    // load ranks
    auto r1_j  = _rank_get(rdict_base, c_j);
    auto r1_i  = _mm512_load_si512(_addr(cache_iter, 2 * cache_step));
    auto r1_k  = _mm512_load_si512(_addr(cache_iter, 4 * cache_step));
    cache_iter = _addr(cache_iter, vector_size);

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

    __mmask16 select_mask;
    // right out the queue (zero part)
    select_mask = _mm512_cmpgt_epi32_mask(r0_j, r0_i);
    select_mask = _mm512_mask_cmpgt_epi32_mask(select_mask, r0_k, r0_j);
    _mm512_mask_compressstoreu_epi32(queue_out0, select_mask, _mm512_add_epi32(r0_j, c_z));
    queue_out0 += _mm_popcnt_u64(select_mask);
    *smask_out0++ = select_mask;

    // right out the queue (ones part)
    select_mask = _mm512_cmpgt_epi32_mask(r1_j, r1_i);
    select_mask = _mm512_mask_cmpgt_epi32_mask(select_mask, r1_k, r1_j);
    _mm512_mask_compressstoreu_epi32(queue_out1, select_mask,  _mm512_add_epi32(r1_j, c_z));
    queue_out1 += _mm_popcnt_u64(select_mask);
    *smask_out1++ = select_mask;

    // calculate the limits from the butterfly
    __m512i min = _mm512_maskz_sub_epi32(_mm512_cmpgt_epi32_mask(n_1w_, n__w0), n_1w_, n__w0);
    __m512i max = _mm512_min_epi32(n_1w_, n__w1);

    // calculate the digit and the base
    auto digit = _mm512_sub_epi32(n_1w1, min);
    auto base  = _mm512_add_epi32(_mm512_sub_epi32(max, min), _mm512_set1_epi32(1));

    // store the limits for the coder
    _mm512_storeu_si512(_addr(coder_iter, 0 * vector_size), digit);
    _mm512_storeu_si512(_addr(coder_iter, 1 * vector_size), base );
    coder_iter = _addr(coder_iter, 2 * vector_size);

    // interleave loads and ranks
    // write out only the used ones (lo part)
    auto mask_lo  = static_cast<std::uint16_t>(*smask_iter++);
    //auto pmask_lo = _compress_mask(mask_lo);
    _interleavelo_compress_and_store(_addr(cache_next, 0 * cache_step), mask_lo, c_i , c_j );
    _interleavelo_compress_and_store(_addr(cache_next, 1 * cache_step), mask_lo, c_j , c_k );
    _interleavelo_compress_and_store(_addr(cache_next, 2 * cache_step), mask_lo, r1_i, r1_j);
    _interleavelo_compress_and_store(_addr(cache_next, 4 * cache_step), mask_lo, r1_j, r1_k);
    cache_next += _mm_popcnt_u32(mask_lo);

    // write out only the used ones (hi part)
    auto mask_hi  = static_cast<std::uint16_t>(*smask_iter++);
    //auto pmask_hi = _compress_mask(mask_hi);
    _interleavehi_compress_and_store(_addr(cache_next, 0 * cache_step), mask_hi, c_i , c_j );
    _interleavehi_compress_and_store(_addr(cache_next, 1 * cache_step), mask_hi, c_j , c_k );
    _interleavehi_compress_and_store(_addr(cache_next, 2 * cache_step), mask_hi, r1_i, r1_j);
    _interleavehi_compress_and_store(_addr(cache_next, 4 * cache_step), mask_hi, r1_j, r1_k);
    cache_next += _mm_popcnt_u32(mask_hi);
  }
  IACA_END
  return 0;
}

int main() {}
