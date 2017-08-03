#g++ -march=skylake-avx512 -std=c++14 -O3 -o ./build/bce decode_main.cpp
#g++ -march=skylake-avx512 -std=c++14 -O3 -o ./build/decode_main.s -S decode_main.cpp

#icpc -xCORE-AVX512 -std=c++14 -O3 -o ./build/bce decode_main.cpp
#icpc -xCORE-AVX512 -std=c++14 -O3 -o ./build/decode_main.s -S decode_main.cpp

icpc -O3 -xCORE-AVX512 -o ./build/bce build/decode_main.s

iaca.sh -graph ./build/graph -trace ./build/trace -arch SKX -o ./build/iaca.txt ./build/bce
pt.py ./build/trace.iacatrace > ./build/trace.txt
dot -Tps ./build/graph1.dot -o ./build/outfile.ps