#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

inline void _permute_and_store(std::uint32_t* dst, __m512i k, __m512i a, __m512i b) {
    _mm512_storeu_si512(dst, _mm512_permutex2var_epi32(a, k, b));
}

inline __m512i _rank_get(__m512i bits, __m512i rank, __m512i indice) {
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

  // cached values (should probably fit L1 or L2)
  std::uint32_t* cache_iter,  // start of the cached values
  std::uint32_t* cache_next,  // where to write the next cached values
  std::size_t    cache_step,  // step between the cache columns in BYTES
                              // columns: c_i, c_k, r1_i, unused/coder, r1_k
                              //   we dont use column 3 to save registers (addressing doesn't allow *3)
                              //   L1/L2 shouldn't be effected by this
                              //   unused memory overhead is not that big
                              //   if we interleave all 8 caches we can use
                              //   that one big free chunk for the coder !

  // the rank dictionary
  std::uint32_t* rdict_base,  // base address of the dictionary
  std::uint32_t  count_zero,  // number of zeros in this dictionary

  // stream of the filter masks
  std::uint16_t* fmask_iter,
  std::uint16_t* fmask_out0,
  std::uint16_t* fmask_out1,

  // where to store the coder information
  std::uint32_t* coder_iter,  // should use the free column in the cache
  __mmask16      k0           // we need the compiler to assume this
                              // might not be -1 but in fact it needs to be
) {
  // constants
  constexpr std::size_t m512i_size = sizeof(__m512i);

  // loop invariants
  auto cX_z = _mm512_set1_epi32(count_zero);  // broadcast the amount of zeros

  // software pipelining preload
  auto cX_j_pipe = _mm512_load_si512(queue_iter   );
  auto offsets   = _mm512_srli_epi32(cX_j_pipe , 5);
  auto bits_pipe = _mm512_i64gather_epi64(offsets, rdict_base + 0, 8);
  auto rank_pipe = _mm512_i64gather_epi64(offsets, rdict_base + 1, 8);

  // this is based on
  //   Computing the longest common prefix array based on the Burrows–Wheeler transform - Uwe Baier
  //     this describes the iteration over the BWT in order to do
  //     a breath-first search on the (virtual) suffix tree
  //   Lossless data compression via substring enumeration - D Dube
  //     this describes the actual compression algorithm
  //     it basically recursively refines a markov model
  //   Improving Compression via Substring Enumeration by Explicit Phase Awareness. - M Béliveau
  //     this explains why the dictionary is actually done
  //     on 8 dictionaries. This implementation uses a wavelet matrix
  //     rather than a regular wavelet tree.

  for (std::size_t i = 0; queue_iter < queue_last; ++i) {
    IACA_START
    // pipeline cX_j
    auto cX_j  = cX_j_pipe;
    cX_j_pipe  = _mm512_load_si512(queue_iter);
    queue_iter = _addr(queue_iter, m512i_size);

    // load from cache (queue and ranks)
    auto cX_i  = _mm512_load_si512(_addr(cache_iter, 0 * cache_step));
    auto cX_k  = _mm512_load_si512(_addr(cache_iter, 1 * cache_step));
    auto r1_i  = _mm512_load_si512(_addr(cache_iter, 2 * cache_step));
    auto r1_k  = _mm512_load_si512(_addr(cache_iter, 4 * cache_step));

    // start preload
    offsets    = _mm512_srli_epi32(cX_j_pipe, 5);
    auto bits  = bits_pipe;
    bits_pipe  = _mm512_mask_i64gather_epi64(bits_pipe, k0, offsets, rdict_base + 0, 8);
    auto rank  = rank_pipe;
    rank_pipe  = _mm512_mask_i64gather_epi64(rank_pipe, k0, offsets, rdict_base + 1, 8);

    // calculate the rank
    auto r1_j  = _rank_get(bits, rank, cX_j);

    // calculate some butterflies vars
    auto n_1w_ = _mm512_sub_epi32(cX_k, cX_j);

    // calculate the number of zeros before each point
    auto r0_i  = _mm512_sub_epi32(cX_i, r1_i);
    auto r0_j  = _mm512_sub_epi32(cX_j, r1_j);
    auto r0_k  = _mm512_sub_epi32(cX_k, r1_k);

    // calculate the rest of the butterfly vars
    auto n__w1 = _mm512_sub_epi32(r1_k, r1_i);
    auto n__w0 = _mm512_sub_epi32(r0_k, r0_i);
    auto n_1w1 = _mm512_sub_epi32(r1_k, r1_j);

    __mmask16 select_mask;
    // write out the queue (zero part)
    select_mask   = _mm512_cmpgt_epi32_mask(r0_j, r0_i);
    select_mask   = _mm512_mask_cmpgt_epi32_mask(select_mask, r0_k, r0_j);
    _mm512_mask_compressstoreu_epi32(queue_out0, select_mask, r0_j);
    queue_out0   += _mm_popcnt_u64(select_mask);
    *fmask_out0++ = select_mask;

    // write out the queue (zero part)
    select_mask   = _mm512_cmpgt_epi32_mask(r1_j, r1_i);
    select_mask   = _mm512_mask_cmpgt_epi32_mask(select_mask, r1_k, r1_j);
    _mm512_mask_compressstoreu_epi32(queue_out1, select_mask, r1_j);
    queue_out1   += _mm_popcnt_u64(select_mask);
    *fmask_out1++ = select_mask;

    // calculate the limits from the butterfly
    __m512i min = _mm512_maskz_sub_epi32(_mm512_cmpgt_epi32_mask(n_1w_, n__w0), n_1w_, n__w0);
    __m512i max = _mm512_min_epi32(n_1w_, n__w1);

    // calculate the digit and the base
    auto digit = _mm512_sub_epi32(n_1w1, min);
    auto base  = _mm512_add_epi32(_mm512_sub_epi32(max, min), _mm512_set1_epi32(1));

    // store the limits for the coder
    _mm512_storeu_si512(_addr(coder_iter, 0 * m512i_size), digit);
    _mm512_storeu_si512(_addr(coder_iter, 1 * m512i_size), base );
    coder_iter = _addr(coder_iter, 2 * m512i_size);

    // interleave loads and ranks
    // write out only the used ones (lo part)
    auto mask_lo  = *fmask_iter++;
    auto shuffle_masklo = _mm512_set_epi32(23, 7,22, 6,21, 5,20, 4,19, 3,18, 2,17, 1,16, 0);
         shuffle_masklo = _mm512_maskz_compress_epi32(mask_lo, shuffle_masklo);
    auto cache_next_tmp = cache_next + _mm_popcnt_u64(mask_lo);
    _permute_and_store(_addr(cache_next, 0 * cache_step), shuffle_masklo, cX_i, cX_j);
    _permute_and_store(_addr(cache_next, 1 * cache_step), shuffle_masklo, cX_j, cX_k);
    _permute_and_store(_addr(cache_next, 2 * cache_step), shuffle_masklo, r1_i, r1_j);
    _permute_and_store(_addr(cache_next, 4 * cache_step), shuffle_masklo, r1_j, r1_k);

    // write out only the used ones (hi part)
    auto mask_hi  = *fmask_iter++;
    auto shuffle_maskhi = _mm512_set_epi32(31,15,30,14,29,13,28,12,27,11,26,10,25, 9,24, 8);
         shuffle_maskhi = _mm512_maskz_compress_epi32(mask_hi, shuffle_maskhi);
    cache_next = cache_next_tmp + _mm_popcnt_u64(mask_hi);
    _permute_and_store(_addr(cache_next_tmp, 0 * cache_step), shuffle_maskhi, cX_i, cX_j);
    _permute_and_store(_addr(cache_next_tmp, 1 * cache_step), shuffle_maskhi, cX_j, cX_k);
    _permute_and_store(_addr(cache_next_tmp, 2 * cache_step), shuffle_maskhi, r1_i, r1_j);
    _permute_and_store(_addr(cache_next_tmp, 4 * cache_step), shuffle_maskhi, r1_j, r1_k);
  }
  IACA_END
  return 0;
}

int main() {}
