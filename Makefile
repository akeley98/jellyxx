FLAGS = -mavx -O3 -Wall -Wextra -DGLM_FORCE_RADIANS=1
CXX = c++ $(FLAGS) -std=c++17 -S
LINKER = c++

jelly++: bin/client.s bin/jellymjf.s
	$(LINKER) bin/client.s bin/jellymjf.s -o jelly++ -lGL -lSDL2

bin/client.s: client.cc beadface.h opengl-functions.inc jellymjf.h
	$(CXX) client.cc -o bin/client.s

bin/jellymjf.s: jellymjf.cc jellymjf.h physics.hpp simd_dvec3.hpp vec3.h
	$(CXX) jellymjf.cc -o bin/jellymjf.s

