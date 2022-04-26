#pragma once
#include <iostream>

#include <npp.h>

#include "helper_cuda.h"

#define NPP_CHECK(status)                                                      \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != NPP_SUCCESS) {                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " NPP failure: " << ret      \
                << " msg:" << _cudaGetErrorEnum(ret) << std::endl;             \
      throw "";                                                                \
    }                                                                          \
  } while (0)
