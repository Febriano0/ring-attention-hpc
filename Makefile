
CXX = g++
MPICXX = mpicxx

CXXFLAGS = -O3 -Wall -Wextra -std=c++17
OMPFLAGS = -fopenmp

BUILD_DIR = build

.PHONY: all setup check clean help

all: help

setup:
	mkdir -p $(BUILD_DIR)

check:
	@echo "Checking development environment..."
	@$(CXX) --version
	@$(MPICXX) --version

help:
	@echo "Ring Attention HPC - Available Commands"
	@echo "make setup  - Create build directory"
	@echo "make check  - Check compiler availability"
	@echo "make clean  - Remove build directory"

clean:
	rm -rf $(BUILD_DIR)
