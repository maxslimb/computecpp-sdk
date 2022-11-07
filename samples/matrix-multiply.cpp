#include <CL/sycl.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

using namespace cl::sycl;

class mxm_kernel;

void display_matrix(float* m, int matSize) {
  if (matSize > 16) {
    return;
  }

  std::cout << "=======" << std::endl;
  for (int i = 0; i < matSize; i++) {
    for (int j = 0; j < matSize; j++) {
      std::cout << m[i * matSize + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "=======" << std::endl;
  ;
}

/* Implements a host C++ version of the matrix multiplication.
 * If compiler supports OpenMP, code is parallelized. Scheduling
 * uses static chunks of block_size. */
void block_host(float* MA, float* MB, float* MC, int matSize) {
  /* We set the block size to 32 for simplicity, though the optimal
   * value will depend on the platform this is run on. */
  int block_size = 32;
  int numBlocks = block_size / matSize;
  int extraBlockLength = block_size % matSize;
  numBlocks = extraBlockLength ? (numBlocks + 1) : (numBlocks);

#pragma omp parallel for num_threads(2) collapse(2)
  for (int bIndexI = 0; bIndexI < matSize; bIndexI += block_size)
    for (int bIndexJ = 0; bIndexJ < matSize; bIndexJ += block_size)
      for (int bIndexK = 0; bIndexK < matSize; bIndexK += block_size) {
        int i = bIndexI;
        int j = bIndexJ;
        int k = bIndexK;
        for (int bi = i; bi < std::min(i + block_size, matSize); bi++)
          for (int bj = j; bj < std::min(j + block_size, matSize); bj++)
            for (int bk = k; bk < std::min(k + block_size, matSize); bk++) {
              MC[bi * matSize + bj] +=
                  MA[bi * matSize + bk] * MB[bk * matSize + bj];
            }
      }
}

/* Obtains the previous power of two from the given integer.
 * It works by masking out all ones after the first one bit,
 * then leaves the first one bit intact, effectively
 * yielding the first power of two < x. */
inline int prevPowerOfTwo(int x) {
  if (x < 0) {
    return 0;
  }
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x - (x >> 1);
}

/* Checks if X is a power of two.
 * If there are bits sets to one after AND with the
 * previous number, then it is not a power of two.
 */
inline bool isPowerOfTwo(int x) { return (x & (x - 1)) == 0; }

/* Function template that performs the matrix * matrix operation. (It is
 * a template because only some OpenCL devices support double-precision
 * floating-point numbers, but it is interesting to make the comparison
 * where available.)
 * Broadly, the function chooses an appropriate work size, then enqueues
 * the matrix * matrix lambda on the queue provided. Because the queues
 * are constructed inside this function, it will block until the work is
 * finished.
 * Note that this example only works for powers of two.
 * */

/* Helper function to indicate the parameters the sample takes. */
void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [omp]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply (minimum 32)"
            << std::endl;
  std::cout << "[omp]    : Run the OpenMP "
            << " Default is to use both " << std::endl;
}

int main(int argc, char* argv[]) {
  float* MA;
  float* MB;
  float* MC;
  bool sycl = true;
  bool omp = true;
  bool error = false;

  if (argc != 2 && argc != 3) {
    usage(argv[0]);
    return 1;
  }

  int matSize = 0;
  try {
    matSize = std::stoi(argv[1]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  if (matSize < 32) {
    usage(argv[0]);
    return 1;
  }

  if (argc == 3) {
    if (std::string(argv[2]) == "omp") {
      omp = true;
    }  else {
      usage(argv[0]);
    }
  }

  MA = new float[matSize * matSize];
  MB = new float[matSize * matSize];
  MC = new float[matSize * matSize];

// Matrix initialization
#pragma omp parallel for collapse(2)
  for (int i = 0; i < matSize; i++)
    for (int j = 0; j < matSize; j++) {
      MA[i * matSize + j] = 0.0f;
      if (i == j) {
        MA[i * matSize + j] = 1.0f;
      }
      MB[i * matSize + j] = 2.0f;
      MC[i * matSize + j] = 0.0f;  // i * matSize + j;
    }

  std::cout << " Input matrix " << std::endl;
  display_matrix(MA, matSize);
  display_matrix(MB, matSize);
  display_matrix(MC, matSize);

  if (omp) {
#if defined(_OPENMP)
    std::cout << "OpenMP: ";
#else
    std::cout << "C++: ";
#endif

    {
      auto start = std::chrono::steady_clock::now();
      block_host(MA, MB, MC, matSize);
      auto end = std::chrono::steady_clock::now();
      auto time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      std::cout << "Time: " << time << std::endl;
      float flops =
          (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
      std::cout << "GFLOPs: " << flops << std::endl;

      bool error = false;
      // Testing
      for (int i = 0; i < matSize; i++)
        for (int j = 0; j < matSize; j++) {
          if (std::fabs(MC[i * matSize + j] - MB[i * matSize + j]) > 1e-8) {
            std::cout << " Position " << i << ", " << j
                      << " differs: " << MC[i * matSize + j]
                      << " != " << MB[i * matSize + j] << std::endl;
            error = true;
          }
        }
      if (!error) {
        std::cout << "Success" << std::endl;
      } else {
        std::cout << " Error in the computation " << std::endl;
      }
    }
  }

 

  delete[] MA;
  delete[] MB;
  delete[] MC;

  return error ? 1 : 0;
}
