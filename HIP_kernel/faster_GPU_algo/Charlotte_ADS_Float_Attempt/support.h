#ifndef ADS_FLOAT_HIP_SUPPORT_H_
#define ADS_FLOAT_HIP_SUPPORT_H_

#include "csr_graph.h"

void parse_args(int argc, char* argv[]);
void output(const CSRGraphTy& graph, const char* output_file);
void usage(const char* program);

#endif  // ADS_FLOAT_HIP_SUPPORT_H_
