CXX=g++
# Use older CUDA to be compatible with Voltas (for development purposes)
CUDA_PATH=/usr/local/cuda-12.9/
NVCC=$(CUDA_PATH)/bin/nvcc
STD=-std=c++20
INCLUDE=./include $(CUDA_PATH)/include ./tests/include
LIB_DIRS=$(CUDA_PATH)/lib64
LIBS=cudart cufft

CXXFLAGS=-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -O3 $(STD) \
	$(addprefix -I, $(INCLUDE))

LDFLAGS= $(addprefix -L, $(LIB_DIRS)) \
	$(addprefix -l, $(LIBS))

NVCCFLAGS=-ccbin $(CXX) $(STD) -O3 --expt-relaxed-constexpr \
	$(addprefix -I, $(INCLUDE)) \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_90,code=sm_90

BUILD_DIR=build

# Targets
RESIDUUM_TESTS_TARGET=$(BUILD_DIR)/residuum-tests
BENCHMARKS_TARGET=$(BUILD_DIR)/benchmarks
SIMULATION_TARGET=$(BUILD_DIR)/sim

# Source lists
CXX_SRC=$(wildcard src/*.cpp)
CU_SRC=$(wildcard src/kernels/*.cu)
RESIDUUM_TESTS_SRC=$(shell find tests/residuum/ -name '*.cpp')
BENCHMARKS_SRC=$(wildcard tests/benchmarks/*.cpp)
SIMULATION_SRC=$(wildcard simulation/*.cpp)

# Object lists
CXX_OBJS=$(CXX_SRC:src/%.cpp=$(BUILD_DIR)/src/%.o)
CU_OBJS=$(CU_SRC:src/kernels/%.cu=$(BUILD_DIR)/src/kernels/%.o)
RESIDUUM_TESTS_OBJS=$(RESIDUUM_TESTS_SRC:tests/%.cpp=$(BUILD_DIR)/tests/%.o)
BENCHMARKS_OBJS=$(BENCHMARKS_SRC:tests/%.cpp=$(BUILD_DIR)/tests/%.o)
SIMULATION_OBJS=$(SIMULATION_SRC:simulation/%.cpp=$(BUILD_DIR)/simulation/%.o)

# Headers
CXX_HEADERS=$(shell find include -name '*.hpp' -o -name '*.h')
CU_HEADERS=$(shell find include -name '*.cuh')
TEST_HEADERS=$(shell find tests/include -name '*.hpp' -o -name '*.h')

.PHONY: all lib residuum-tests benchmarks simulation clear clean

all: lib residuum-tests benchmarks simulation

lib: $(CXX_OBJS) $(CU_OBJS)

residuum-tests: $(RESIDUUM_TESTS_TARGET)

benchmarks: $(BENCHMARKS_TARGET)

simulation: $(SIMULATION_TARGET)

# Link residuum tests
$(RESIDUUM_TESTS_TARGET): $(CXX_OBJS) $(CU_OBJS) $(RESIDUUM_TESTS_OBJS)
	@mkdir -p $(dir $@)
	@echo "Linking residuum tests: $@"
	@$(CXX) $(CXXFLAGS) $(CXX_OBJS) $(CU_OBJS) $(RESIDUUM_TESTS_OBJS) $(LDFLAGS) -o $@

# Link benchmarks
$(BENCHMARKS_TARGET): $(CXX_OBJS) $(CU_OBJS) $(BENCHMARKS_OBJS)
	@mkdir -p $(dir $@)
	@echo "Linking benchmarks: $@"
	@$(CXX) $(CXXFLAGS) $(CXX_OBJS) $(CU_OBJS) $(BENCHMARKS_OBJS) $(LDFLAGS) -o $@

# Link simulation
$(SIMULATION_TARGET): $(CXX_OBJS) $(CU_OBJS) $(SIMULATION_OBJS)
	@mkdir -p $(dir $@)
	@echo "Linking simulation: $@"
	@$(CXX) $(CXXFLAGS) $(CXX_OBJS) $(LIB_CU_OBJS) $(SIMULATION_OBJS) $(LDFLAGS) -o $@

# Compile residuum tests
$(BUILD_DIR)/tests/residuum/%.o: tests/residuum/%.cpp $(CXX_HEADERS) $(TEST_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling residuum tests: $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile benchmarks
$(BUILD_DIR)/tests/benchmarks/%.o: tests/benchmarks/%.cpp $(CXX_HEADERS) $(TEST_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling benchmarks: $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile simulation
$(SIMULATION_OBJS): $(SIMULATION_SRC) $(CXX_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling simulation: $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile host (C++)
$(BUILD_DIR)/src/%.o: src/%.cpp $(CXX_HEADERS) $(CU_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling host: $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile device (CUDA)
$(BUILD_DIR)/src/kernels/%.o: src/kernels/%.cu $(CU_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling device: $<"
	@$(NVCC) $(NVCCFLAGS) --compile -cudart static $< -o $@

clear:
	@echo "Removing build directory..."
	@rm -rf $(BUILD_DIR)

clean: clear

