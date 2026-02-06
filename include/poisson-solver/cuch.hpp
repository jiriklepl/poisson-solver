// Written by Martin Kruliš for CUDA Blur Stencil lab assignment
// Modified to also contain checks for cuFFT

#ifndef CUCH_HPP
#define CUCH_HPP

#include <cuda_runtime.h>
#include <cufft.h>
#include <sstream>
#include <stdexcept>

namespace poisson_solver {
/**
 * A stream exception that is base for all runtime errors.
 */
class CudaError : public std::exception {
protected:
    std::string mMessage;  ///< Internal buffer where the message is kept.
    cudaError_t mStatus;

public:
    CudaError(cudaError_t status = cudaSuccess)
        : std::exception(), mStatus(status) {}
    CudaError(const char *msg, cudaError_t status = cudaSuccess)
        : std::exception(), mMessage(msg), mStatus(status) {}
    CudaError(const std::string &msg, cudaError_t status = cudaSuccess)
        : std::exception(), mMessage(msg), mStatus(status) {}
    virtual ~CudaError() throw() {}

    virtual const char *what() const throw() { return mMessage.c_str(); }

    // Overloading << operator that uses stringstream to append data to
    // mMessage.
    template <typename T>
    CudaError &operator<<(const T &data) {
        std::stringstream stream;
        stream << mMessage << data;
        mMessage = stream.str();
        return *this;
    }
};

/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile,
                        const char *errMsg = NULL) {
    if (status != cudaSuccess) {
        throw(CudaError(status)
              << "CUDA Error (" << status << "): " << cudaGetErrorString(status)
              << "\n"
              << "at " << srcFile << "[" << line << "]: " << errMsg);
    }
}

static const char *cufftGetErrorString(cufftResult status) {
    switch (status) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
        case CUFFT_MISSING_DEPENDENCY:
            return "CUFFT_MISSING_DEPENDENCY";
        case CUFFT_NVRTC_FAILURE:
            return "CUFFT_NVRTC_FAILURE";
        case CUFFT_NVJITLINK_FAILURE:
            return "CUFFT_NVJITLINK_FAILURE";
        case CUFFT_NVSHMEM_FAILURE:
            return "CUFFT_NVSHMEM_FAILURE";
    }
    return "Unknown cufftResult";
}

// cufft alternative to the _cuda_check.
inline void _cufft_check(cufftResult status, int line, const char *src_file,
                         const char *err_msg = NULL) {
    if (status != CUFFT_SUCCESS) {
        std::stringstream stream;
        stream << "CUFFT Error (" << status
               << "): " << cufftGetErrorString(status) << "\n"
               << "at " << src_file << "[" << line << "]: " << err_msg;

        throw std::runtime_error(stream.str());
    }
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)

#define CUFFTCH(status) _cufft_check(status, __LINE__, __FILE__, #status)
}  // namespace poisson_solver

#endif  // CUCH_HPP
