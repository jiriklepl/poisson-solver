CPP=g++
STD=-std=c++20
CFLAGS=-Wall -Wextra -Wno-unused-parameter -O3 $(STD) -Iinclude

BUILD_DIR=build
TARGET=$(BUILD_DIR)/test-runner

LIB_SRC=$(wildcard src/*.cpp)
TEST_SRC=$(wildcard tests/*.cpp)

LIB_OBJS=$(LIB_SRC:src/%.cpp=$(BUILD_DIR)/src/%.o)
TEST_OBJS=$(TEST_SRC:tests/%.cpp=$(BUILD_DIR)/tests/%.o)

HEADERS=$(shell find include -name '*.hpp')

.PHONY: all lib tests clear clean

all: tests

lib: $(LIB_OBJS)

tests: $(TARGET)

$(TARGET): $(LIB_OBJS) $(TEST_OBJS)
	@mkdir -p $(dir $@)
	@echo "Linking executable: $@"
	$(CPP) $(CFLAGS) $(LIB_OBJS) $(TEST_OBJS) -o $@

$(BUILD_DIR)/src/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling Lib: $<"
	$(CPP) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	@echo "Compiling Test: $<"
	$(CPP) $(CFLAGS) -c $< -o $@

clear:
	@echo "Removing build directory..."
	@rm -rf $(BUILD_DIR)

clean: clear

