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
  // the last element MUST be bigger than the first because 16 contexts don't fit within 32 bits
  const     __m512i     permt_left = _mm512_set_epi32(14,13,12,11,10, 9, 8 , 7, 6, 5, 4, 3, 2, 1, 0,15);

  // loop invariants
  auto cX_z = _mm512_set1_epi32(count_zero);  // broadcast the amount of zeros
  
  // init ANS decoder
  auto state_sml0 = _mm512_set1_epi32(0);  // actually loaded from stream_ptr
  auto state_sml1 = _mm512_set1_epi32(0);  // actually loaded from stream_ptr
  auto state_sml2 = _mm512_set1_epi32(0);  // actually loaded from stream_ptr

  // @todo software pipelining
  
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

  // decompressing is a lot more complex
  // we aim for around 40-50 cycles here:
  // 21 loads + 
  while (queue_iter < queue_last) {
    //IACA_START
    // cX_j
    auto cX_j  = _mm512_load_si512(queue_iter);
    queue_iter = _addr(queue_iter, m512i_size);
    
    // gather the bits
    auto offsets = _mm512_srli_epi32(cX_j, 5);
    auto indice  = _mm512_and_epi32 (cX_j, _mm512_set1_epi32(0x1F));
    auto bits    = _mm512_i32gather_epi32(offsets, rdict_base + 0, 8);

    // load from cache (queue and ranks)
    auto cX_i  = _mm512_load_si512(_addr(cache_iter, 0 * cache_step));
    auto cX_k  = _mm512_load_si512(_addr(cache_iter, 1 * cache_step));
    auto r1_i  = _mm512_load_si512(_addr(cache_iter, 2 * cache_step));
    auto r1_k  = _mm512_load_si512(_addr(cache_iter, 4 * cache_step));
    
    // calculate some butterflies vars
    auto n_1w_ = _mm512_sub_epi32(cX_k, cX_j);

    // calculate the number of zeros before each point
    auto r0_i  = _mm512_sub_epi32(cX_i, r1_i);
    auto r0_k  = _mm512_sub_epi32(cX_k, r1_k);

    // calculate more of the butterfly vars
    auto n__w1 = _mm512_sub_epi32(r1_k, r1_i);
    auto n__w0 = _mm512_sub_epi32(r0_k, r0_i);

    // calculate the limits from the butterfly
    __m512i min = _mm512_maskz_sub_epi32(_mm512_cmpgt_epi32_mask(n_1w_, n__w0), n_1w_, n__w0);
    __m512i max = _mm512_min_epi32(n_1w_, n__w1);

    // calculate the digit and the base
    auto base  = _mm512_add_epi32(_mm512_sub_epi32(max, min), _mm512_set1_epi32(1));
    auto digit =  decode_ans_int (state_sml0, base, stream_ptr);
    
    // calculate the rest of the butterfly vars
    auto n_1w1 = _mm512_add_epi32(min  , digit);
    auto n_0w1 = _mm512_sub_epi32(n__w1, n_1w1);
    auto n_1w0 = _mm512_sub_epi32(n_1w_, n_1w1);
    
    // calculate r0_j and r1_j
    auto r1_j  = _mm512_sub_epi32(r1_k, n_1w1);
    auto r0_j  = _mm512_sub_epi32(cX_j, r1_j );
      
    // calculate set and clear maks
    // @todo validate this. It assumes full 32 bit unsigned shift indices
    //       according to Intels Intrinsic Guide thats correct
    auto index_mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), indice);
    auto clear_mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_add_epi32(indice, n_0w1));
    auto set___mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), _mm512_sub_epi32(indice, n_1w0));
    clear_mask = _mm512_sub_epi32(clear_mask, index_mask);
    set___mask = _mm512_sub_epi32(index_mask, set___mask);
    // @todo if it turns out that there are a lot of conflicts
    //       then we could theoretically merge the clear and set masks in log
    //       time and apply them only once.
    //       Even simple linear time merging will proably pays off by
    //       avoiding multiple scatters.
    
    
    // update the rank dictionary
    __mmask16 mask = k0;
    while (1) {
      // conflict detection is too slow on skylake-avx512
      auto conf_free = _mm512_mask_cmp_epi32_mask  (mask     , offsets, _mm512_permutevar_epi32(permt_left, offsets), 4/*_MM_CMPINT_NEQ*/);
      auto rank_mask = _mm512_mask_cmpeq_epi32_mask(conf_free, offsets, _mm512_srli_epi32(cX_i, 5));
      
      // clear and set the bits
      bits = _mm512_andnot_si512(clear_mask, bits);  // bits &= ~clear_mask
      bits = _mm512_or_si512    (set___mask, bits);  // bits |=  set___mask
      
      // calcualte the ranks
      auto prev_bits = _mm512_min_epi32(indice, n_1w0  );  // part of n_1w0 that actually is within this chunk
      auto rank      = _mm512_sub_epi32(r1_j, prev_bits);  // one those decrease the rank
      
      // scatter updates for bits and rank
      _mm512_mask_i32scatter_epi32 (rdict_base + 0, conf_free, offsets, bits, 8);
      _mm512_mask_i32scatter_epi32 (rdict_base + 1, rank_mask, offsets, rank, 8);
      
      mask = _mm512_kandn(conf_free, mask);  // drop the updated / conflict free elements
      if (_mm512_kortestz(mask, mask)) // we're done
        break;
      
      // regather bits by rotating the updated values to the right
      bits = _mm512_permutevar_epi32(permt_left, bits);
    }
    
    __mmask16 filter_mask;
    // write out the queue (zero part)
    filter_mask   = _mm512_cmpgt_epi32_mask(r0_j, r0_i);
    filter_mask   = _mm512_mask_cmpgt_epi32_mask(filter_mask, r0_k, r0_j);
    _mm512_mask_compressstoreu_epi32(queue_out0, filter_mask, r0_j);
    queue_out0   += _mm_popcnt_u64(filter_mask);
    *fmask_out0++ = filter_mask;

    // write out the queue (zero part)
    filter_mask   = _mm512_cmpgt_epi32_mask(r1_j, r1_i);
    filter_mask   = _mm512_mask_cmpgt_epi32_mask(filter_mask, r1_k, r1_j);
    _mm512_mask_compressstoreu_epi32(queue_out1, filter_mask, r1_j);
    queue_out1   += _mm_popcnt_u64(filter_mask);
    *fmask_out1++ = filter_mask;

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
  //IACA_END
}

#if 0
int main() {}
#endif
