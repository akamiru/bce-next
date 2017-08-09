#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

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

template<class T>
inline T* _addr(T* base, std::size_t byte_step) {
  return reinterpret_cast<T*>(
    reinterpret_cast<std::uint8_t*>(base) + byte_step
  );
}

void cache_fill (
  // The queues
  //  current
  std::uint32_t* queue_iter,  // start of the queue
  std::uint32_t* queue_last,  // end   of the queue
  std::size_t    queue_step,  // step  of the queue

  // cached values (should probably fit L1 or L2)
  std::uint32_t* cache_iter,  // start of the cached values
  std::size_t    cache_step,  // step between the cache columns in BYTES
                              // columns: c_i, c_k, r1_i, unused/coder, r1_k
                              //   we dont use column 3 to save registers (addressing doesn't allow *3)
                              //   L1/L2 shouldn't be effected by this
                              //   unused memory overhead is not that big
                              //   if we interleave all 8 caches we can use
                              //   that one big free chunk for the coder !

  // the rank dictionary
  std::uint32_t* rdict_base,
  __mmask16      k0           // we need the compiler to assume this
                              // might not be -1 but in fact it needs to be
) {
  // constants
  constexpr std::size_t m512i_size = sizeof(__m512i);
  const __m512i select_bits = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10, 8, 6, 4, 2, 0);  // lo address
  const __m512i select_rank = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11, 9, 7, 5, 3, 1);  // hi address

  // Fill the cache
  while (queue_iter < queue_last) {
    IACA_START
    // load from queue
    auto cX_i  = _mm512_load_si512(_addr(queue_iter, 0 * queue_step));
    auto cX_k  = _mm512_load_si512(_addr(queue_iter, 2 * queue_step));
    queue_iter = _addr(queue_iter, m512i_size);

    // gather values: cX_i
    auto offs_i    = _mm512_srli_epi32(cX_i, 5);
    auto data_lo_i = _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), k0, _mm512_castsi512_si256   (offs_i   ), rdict_base, 8);
    auto data_hi_i = _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), k0, _mm512_extracti32x8_epi32(offs_i, 1), rdict_base, 8);

    // gather values: cX_k
    auto offs_k    = _mm512_srli_epi32(cX_k, 5);
    auto data_lo_k = _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), k0, _mm512_castsi512_si256   (offs_k   ), rdict_base, 8);
    auto data_hi_k = _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), k0, _mm512_extracti32x8_epi32(offs_k, 1), rdict_base, 8);

    // permute
    auto bits_i = _mm512_permutex2var_epi32(data_lo_i, select_bits, data_hi_i);
    auto rank_i = _mm512_permutex2var_epi32(data_lo_i, select_rank, data_hi_i);
    auto bits_k = _mm512_permutex2var_epi32(data_lo_k, select_bits, data_hi_k);
    auto rank_k = _mm512_permutex2var_epi32(data_lo_k, select_rank, data_hi_k);

    // fill the cache
    _mm512_store_si512(_addr(cache_iter, 0 * cache_step), cX_i);
    _mm512_store_si512(_addr(cache_iter, 1 * cache_step), cX_k);
    _mm512_store_si512(_addr(cache_iter, 2 * cache_step), _rank_get(bits_i, rank_i, cX_i));
    _mm512_store_si512(_addr(cache_iter, 4 * cache_step), _rank_get(bits_k, rank_k, cX_k));
    cache_iter = _addr(cache_iter, m512i_size);
  }
  IACA_END
}

int main() {}
