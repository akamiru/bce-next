#include <cstdint>
#include "immintrin.h"

#include "/home/christoph/intel/iaca/include/iacaMarks.h"

#include "decode_ans.cpp"

inline void _permute_and_store(std::uint32_t* dst, __m512i k, __m512i a, __m512i b) {
    _mm512_storeu_si512(dst, _mm512_permutex2var_epi32(a, k, b));
}

#if 0
template<class T>
inline T* _addr(T* base, std::size_t byte_step) {
  return reinterpret_cast<T*>(
    reinterpret_cast<std::uint8_t*>(base) + byte_step
  );
}
#endif

void decode_main (
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
  std::uint8_t*  stream_ptr,  // where the ANS decoder gets its bytes from
  __mmask16      k0           // we need the compiler to assume this
                              // might not be -1 but in fact it needs to be
) {
  // constants
  constexpr std::size_t m512i_size = sizeof(__m512i);
  const __m512i select_bits = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10, 8, 6, 4, 2, 0);  // lo address
  const __m512i select_rank = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11, 9, 7, 5, 3, 1);  // hi address
  const __m512i select_lo   = _mm512_set_epi32(23, 7,22, 6,21, 5,20, 4,19, 3,18, 2,17, 1,16, 0);  // lo address inverse
  const __m512i select_hi   = _mm512_set_epi32(31,15,30,14,29,13,28,12,27,11,26,10,25, 9,24, 8);  // hi address inverse
  const __m512i select_last = _mm512_set_epi64(7,7,7,7,7,7,7,7);  // broadcast the 64 bit last to all others
  
  // lambdas
  const auto rotate_l1 = [](auto i) { return _mm512_permutexvar_epi32(
    _mm512_set_epi32(14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,15), i);
  };
  const auto rotate_l2 = [](auto i) { return _mm512_permutexvar_epi32(
    _mm512_set_epi32(13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,15,14), i);
  };
  const auto rotate_l4 = [](auto i) { return _mm512_permutexvar_epi32(
    _mm512_set_epi32(11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,15,14,13,12), i);
  };
  const auto rotate_l8 = [](auto i) { return _mm512_permutexvar_epi32(
    _mm512_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0,15,14,13,12,11,10, 9, 8), i);
  };

  // loop invariants
  auto cX_z = _mm512_set1_epi32(count_zero);  // broadcast the amount of zeros

  // init ANS decoder
  auto state_sml0 = _mm512_set1_epi32(0);  // actually loaded from stream_ptr
  auto state_sml1 = _mm512_set1_epi32(0);  // actually loaded from stream_ptr
  auto state_sml2 = _mm512_set1_epi32(0);  // actually loaded from stream_ptr
  // auto state_sml0 = _mm512_set1_ps(0);  // actually loaded from stream_ptr
  // auto state_sml1 = _mm512_set1_ps(0);  // actually loaded from stream_ptr
  // auto state_sml2 = _mm512_set1_ps(0);  // actually loaded from stream_ptr

  // @todo software pipelining
  //auto cX_j_pipe = _mm512_load_si512(queue_iter   );
  //auto offsets   = _mm512_srli_epi32(cX_j_pipe , 5);
  //auto bits_pipe = _mm512_i64gather_epi64(offsets, rdict_base + 0, 8);
  //auto rank_pipe = _mm512_i64gather_epi64(offsets, rdict_base + 1, 8);
  
  // bypassing
  auto bits_bplo = _mm512_set1_epi32(0);
  auto bits_bphi = _mm512_set1_epi32(0);

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

  // Sadly decompressing is a lot more complex
  while (queue_iter < queue_last) {
    IACA_START
    // load cX_j (ends +4)
    auto cX_j  = _mm512_load_si512(queue_iter);
    queue_iter = _addr(queue_iter, m512i_size);

    // load from cache (queue and ranks) (ends +7)
    auto cX_i  = _mm512_load_si512(_addr(cache_iter, 0 * cache_step));
    auto cX_k  = _mm512_load_si512(_addr(cache_iter, 1 * cache_step));
    auto r1_i  = _mm512_load_si512(_addr(cache_iter, 2 * cache_step));
    auto r1_k  = _mm512_load_si512(_addr(cache_iter, 4 * cache_step));
    cache_iter = _addr(cache_iter, m512i_size);

    // gather the bits and rank (ends +46)
    auto offsets = _mm512_srli_epi32(cX_j, 5);
    auto indice  = _mm512_and_epi32 (cX_j, _mm512_set1_epi32(0x1F));
    auto gather_lo = _mm512_cmpgt_epi32_mask(offsets, bits_bplo);  // bits_bplo contains the previous offset
    auto gather_hi = gather_lo >> 8;                               // bits_bphi contains the previous bits
    bits_bplo    = _mm512_mask_i32gather_epi64(bits_bphi, gather_lo,_mm512_castsi512_si256   (offsets   ), rdict_base, 8);
    bits_bphi    = _mm512_mask_i32gather_epi64(bits_bphi, gather_hi,_mm512_extracti32x8_epi32(offsets, 1), rdict_base, 8);
    auto bits    = _mm512_permutex2var_epi32(bits_bplo, select_bits, bits_bphi);
    auto rank    = _mm512_permutex2var_epi32(bits_bplo, select_rank, bits_bphi);

    // calculate some butterflies vars (ends +7)
    auto n_1w_ = _mm512_sub_epi32(cX_k, cX_j);

    // calculate the number of zeros before each point (ends +8)
    auto r0_i  = _mm512_sub_epi32(cX_i, r1_i);
    auto r0_k  = _mm512_sub_epi32(cX_k, r1_k);

    // calculate more of the butterfly vars (ends +9)
    auto n__w1 = _mm512_sub_epi32(r1_k, r1_i);
    auto n__w0 = _mm512_sub_epi32(r0_k, r0_i);

    // calculate the limits from the butterfly (ends +12)
    auto min_mask = _mm512_cmpgt_epi32_mask(n_1w_, n__w0);
    auto min      = _mm512_sub_epi32(n_1w_, n__w0);
    auto max      = _mm512_min_epi32(n_1w_, n__w1);

    // calculate the digit and the base (ends +14 | +34 assuming 20 cycle ANS latency)
    auto base  = _mm512_add_epi32(_mm512_mask_sub_epi32(max, min_mask, max, min), _mm512_set1_epi32(1));
    auto digit =  decode_ans_int (state_sml0, base, stream_ptr);
    auto tmp   = state_sml0;
    state_sml0 = state_sml1;
    state_sml1 = state_sml2;
    state_sml2 = tmp;

    // calculate the rest of the butterfly vars (ends + 36)
    auto n_1w1 = _mm512_mask_add_epi32(digit, min_mask, digit, min);
    auto n_0w1 = _mm512_sub_epi32(n__w1, n_1w1);
    auto n_1w0 = _mm512_sub_epi32(n_1w_, n_1w1);

    // calculate r0_j and r1_j (ends + 37)
    auto r1_j  = _mm512_sub_epi32(r1_k, n_1w1);
    auto r0_j  = _mm512_sub_epi32(cX_j, r1_j );
    
    // calcualte the ranks (ends +47)
    auto rank_mask   = _mm512_cmpeq_epi32_mask(offsets, _mm512_srli_epi32(cX_i, 5));
    auto prev_bits   = _mm512_min_epi32(indice, n_1w0);                          // part of n_1w0 that actually is within this chunk
         rank        = _mm512_mask_sub_epi32(rank, rank_mask, r1_j, prev_bits);  // the ones which actually decrease the rank
    auto conflicts_r = _mm512_mask_cmp_epi32_mask(0xFFFE, offsets, rotate_l1(offsets), 4);

    __mmask16 filter_mask;
    // write out the queue (ones part) (ends +48)
    filter_mask   = _mm512_cmpgt_epi32_mask(r1_j, r1_i);
    filter_mask   = _mm512_mask_cmpgt_epi32_mask(filter_mask, r1_k, r1_j);
    _mm512_mask_compressstoreu_epi32(queue_out1, filter_mask, r1_j);
    queue_out1   += _mm_popcnt_u64(filter_mask);
    *fmask_out1++ = filter_mask;
    
    // write out the queue (zero part) (ends +49)
    filter_mask   = _mm512_cmpgt_epi32_mask(r0_j, r0_i);
    filter_mask   = _mm512_mask_cmpgt_epi32_mask(filter_mask, r0_k, r0_j);
    _mm512_mask_compressstoreu_epi32(queue_out0, filter_mask, r0_j);
    queue_out0   += _mm_popcnt_u64(filter_mask);
    *fmask_out0++ = filter_mask;

    // interleave loads and ranks
    // write out only the used ones (lo part) (ends +45)
    auto mask_lo  = *fmask_iter++;
    auto shuffle_masklo = _mm512_maskz_compress_epi32(mask_lo, select_lo);
    auto cache_next_tmp = cache_next + _mm_popcnt_u64(mask_lo);
    _permute_and_store(_addr(cache_next, 0 * cache_step), shuffle_masklo, cX_i, cX_j);  // ends at ~22
    _permute_and_store(_addr(cache_next, 1 * cache_step), shuffle_masklo, cX_j, cX_k);  // ends at ~23
    _permute_and_store(_addr(cache_next, 2 * cache_step), shuffle_masklo, r1_i, r1_j);  // ends at +44
    _permute_and_store(_addr(cache_next, 4 * cache_step), shuffle_masklo, r1_j, r1_k);  // ends at +45

    // write out only the used ones (hi part) (ends +47)
    auto mask_hi  = *fmask_iter++;
    auto shuffle_maskhi = _mm512_maskz_compress_epi32(mask_hi, select_hi);
    cache_next = cache_next_tmp + _mm_popcnt_u64(mask_hi);
    _permute_and_store(_addr(cache_next_tmp, 0 * cache_step), shuffle_maskhi, cX_i, cX_j);  // ends at ~24
    _permute_and_store(_addr(cache_next_tmp, 1 * cache_step), shuffle_maskhi, cX_j, cX_k);  // ends at ~25
    _permute_and_store(_addr(cache_next_tmp, 2 * cache_step), shuffle_maskhi, r1_i, r1_j);  // ends at +46
    _permute_and_store(_addr(cache_next_tmp, 4 * cache_step), shuffle_maskhi, r1_j, r1_k);  // ends at +47
    
    
    
    // second stage
    // pipeline z: offsets, indice, n_0w1, n_1w0, bits, rank
    // pipeline k: rank_mask, conflicts_r
    
    
    
    // calculate set and clear maks (ends +4)
    // @todo validate this. It assumes full 32 bit unsigned shift indices
    //       according to Intels Intrinsic Guide thats correct
    auto indx_mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), indice);
    auto drop_mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_add_epi32(indice, n_0w1));
    auto set__mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_sub_epi32(indice, n_1w0));
    drop_mask = _mm512_sub_epi32(drop_mask, indx_mask);
    set__mask = _mm512_sub_epi32(indx_mask, set__mask);

    // conflict masks
    auto conflicts_1 = conflicts_r & (conflicts_r << 1);  // ends +5
    auto conflicts_2 = conflicts_1 & (conflicts_1 << 2);  // ends +10
    auto conflicts_4 = conflicts_2 & (conflicts_2 << 4);  // ends +15
    // resolve conflicts
    drop_mask = _mm512_mask_or_epi32(drop_mask, conflicts_r, drop_mask, rotate_l1(drop_mask));  // ends +8
    set__mask = _mm512_mask_or_epi32(set__mask, conflicts_r, set__mask, rotate_l1(set__mask));  // ends +9
    drop_mask = _mm512_mask_or_epi32(drop_mask, conflicts_1, drop_mask, rotate_l2(drop_mask));  // ends +12
    set__mask = _mm512_mask_or_epi32(set__mask, conflicts_1, set__mask, rotate_l2(set__mask));  // ends +13
    drop_mask = _mm512_mask_or_epi32(drop_mask, conflicts_2, drop_mask, rotate_l4(drop_mask));  // ends +16
    set__mask = _mm512_mask_or_epi32(set__mask, conflicts_2, set__mask, rotate_l4(set__mask));  // ends +17
    drop_mask = _mm512_mask_or_epi32(drop_mask, conflicts_4, drop_mask, rotate_l8(drop_mask));  // ends +20
    set__mask = _mm512_mask_or_epi32(set__mask, conflicts_4, set__mask, rotate_l8(set__mask));  // ends +21

    // apply masks - truth table:
    // a | 1 1 1 1 0 0 0 0 - input
    // b | 1 1 0 0 1 1 0 0 - drop
    // c | 1 0 1 0 1 0 1 0 - set
    // --|-------------------------
    // d | 1 0 1 1 0 0 1 0 - result
    bits = _mm512_ternarylogic_epi32(bits, drop_mask, set__mask, 0xB2);  // ends +22

    // scatter updates for bits and rank
    bits_bplo = _mm512_permutex2var_epi32(bits, select_lo, rank);  // ends +25
    bits_bphi = _mm512_permutex2var_epi32(bits, select_hi, rank);  // ends +26
    _mm512_mask_i32scatter_epi64(rdict_base, ~(conflicts_r >> 1), _mm512_castsi512_si256   (offsets   ), bits_bplo, 8);  // ends +43
    _mm512_mask_i32scatter_epi64(rdict_base, ~(conflicts_r >> 1), _mm512_extracti32x8_epi32(offsets, 1), bits_bphi, 8);  // ends +51
    bits_bplo = _mm512_permutexvar_epi64(select_last, offsets  );  // ends +3
    bits_bphi = _mm512_permutexvar_epi64(select_last, bits_bphi);  // ends +29
  }
  IACA_END
}

#if 0
int main() {}
#endif
