#pragma once
#pragma warning(disable: 4819)

// copied from http://moznion.hatenadiary.com/entry/2014/09/11/113406
// Thanks.
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         std::system("pause"); \
         exit(err); \
     } \
} while(0)

void set_constant_var(const float* const c);
void call_kernel_func_gpu(const float* const cloud_dev, char* const is_visible_dev, const float lambda_sqrd, const int i, const int numel);
