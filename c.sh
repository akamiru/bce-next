#g++ -march=skylake-avx512 -std=c++14 -O3 -o ./build/bce main.cpp
#g++ -march=skylake-avx512 -std=c++14 -O3 -o ./build/main.s -S main.cpp

#icpc -xCORE-AVX512 -std=c++14 -O3 -o ./build/bce main.cpp
#icpc -xCORE-AVX512 -std=c++14 -O3 -o ./build/main.s -S main.cpp

icpc -O3 -xCORE-AVX512 -o ./build/bce build/main.s

iaca.sh -graph ./build/graph -trace ./build/trace -arch SKX -o ./build/iaca.txt ./build/bce
pt.py ./build/trace.iacatrace > ./build/trace.txt
dot -Tps ./build/graph1.dot -o ./build/outfile.ps