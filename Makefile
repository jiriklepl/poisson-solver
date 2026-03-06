CPP=g++
NVCC=nvcc
STD=-std=c++20
INCLUDE=./include /usr/local/cuda/include ./vendor
LIB_DIRS=/usr/local/cuda/lib64
LIBS=cudart cufft

CFLAGS=-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -O3 $(STD) \
	$(addprefix -I, $(INCLUDE))

LDFLAGS= $(addprefix -L, $(LIB_DIRS)) \
	$(addprefix -l, $(LIBS))

NVCCFLAGS=-ccbin $(CPP) $(STD) -O3 --expt-relaxed-constexpr \
	$(addprefix -I, $(INCLUDE)) \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_90,code=sm_90

BUILD_DIR=build
TARGET=$(BUILD_DIR)/test-runner

#Simulation config
SIM_TARGET=$(BUILD_DIR)/sim
SIM_SRC=simulation/simulation.cpp
SIM_OBJ=$(BUILD_DIR)/simulation/simulation.o

#Source lists
LIB_CPP_SRC=$(wildcard src/*.cpp)
LIB_CU_SRC=$(wildcard src/kernels/*.cu)
TEST_SRC=$(wildcard tests/*.cpp)

# Object lists
LIB_CPP_OBJS=$(LIB_CPP_SRC:src/%.cpp=$(BUILD_DIR)/src/%.o)
LIB_CU_OBJS=$(LIB_CU_SRC:src/kernels/%.cu=$(BUILD_DIR)/src/kernels/%.o)
TEST_OBJS=$(TEST_SRC:tests/%.cpp=$(BUILD_DIR)/tests/%.o)

# Headers
HEADERS=$(shell find include -name '*.hpp' -o -name '*.h')
CU_HEADERS=$(shell find include -name '*.cuh')

.PHONY: all lib tests sim clear clean

all: tests

lib: $(LIB_CPP_OBJS) $(LIB_CU_OBJS)

sim: $(SIM_TARGET)

tests: $(TARGET)

# Link executable
$(TARGET): $(LIB_CPP_OBJS) $(LIB_CU_OBJS) $(TEST_OBJS)
	@mkdir -p $(dir $@)
	@echo "Linking executable: $@"
	$(CPP) $(CFLAGS) $(LIB_CPP_OBJS) $(LIB_CU_OBJS) $(TEST_OBJS) $(LDFLAGS) -o $@

# Link simulation
$(SIM_TARGET): $(LIB_CPP_OBJS) $(LIB_CU_OBJS) $(SIM_OBJ)
	@mkdir -p $(dir $@)
	@echo "Linking simulation: $@"
	$(CPP) $(CFLAGS) $(LIB_CPP_OBJS) $(LIB_CU_OBJS) $(SIM_OBJ) $(LDFLAGS) -o $@

# Compile host (C++)
$(BUILD_DIR)/src/%.o: src/%.cpp $(HEADERS) $(CU_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling host: $<"
	$(CPP) $(CFLAGS) -c $< -o $@

# Compile device (CUDA)
$(BUILD_DIR)/src/kernels/%.o: src/kernels/%.cu $(CU_HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling device: $<"
	$(NVCC) $(NVCCFLAGS) --compile -cudart static $< -o $@

# Compile tests
$(BUILD_DIR)/tests/%.o: tests/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling tests: $<"
	$(CPP) $(CFLAGS) -c $< -o $@

# Compile simulation
$(SIM_OBJ): $(SIM_SRC) $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling simulation: $<"
	$(CPP) $(CFLAGS) -c $< -o $@

clear:
	@echo "Removing build directory..."
	@rm -rf $(BUILD_DIR)

clean: clear

