#include "support.h"

#include <getopt.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern int start_node;
extern char* INPUT;
extern char* OUTPUT;
extern int HIP_DEVICE;

void usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s [-q] [-g gpu-number] [-s start-node] "
                 "[-o output-file] graph-file\n",
                 program);
}

void parse_args(int argc, char* argv[]) {
    int option;
    while ((option = getopt(argc, argv, "g:qo:s:")) != -1) {
        switch (option) {
            case 'q':
                break;  // Retained for command-line compatibility.
            case 'o':
                OUTPUT = optarg;
                break;
            case 'g': {
                char* end = nullptr;
                errno = 0;
                const long value = std::strtol(optarg, &end, 10);
                if (errno != 0 || end == optarg || *end != '\0' || value < 0 ||
                    value > INT_MAX) {
                    std::fprintf(stderr, "Invalid GPU device '%s'\n", optarg);
                    std::exit(EXIT_FAILURE);
                }
                HIP_DEVICE = static_cast<int>(value);
                break;
            }
            case 's': {
                char* end = nullptr;
                errno = 0;
                const long value = std::strtol(optarg, &end, 10);
                if (errno != 0 || end == optarg || *end != '\0' || value < 0 ||
                    value > INT_MAX) {
                    std::fprintf(stderr, "Invalid source vertex '%s'\n",
                                 optarg);
                    std::exit(EXIT_FAILURE);
                }
                start_node = static_cast<int>(value);
                break;
            }
            default:
                usage(argv[0]);
                std::exit(EXIT_FAILURE);
        }
    }

    if (optind >= argc) {
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    INPUT = argv[optind];
    if (optind + 1 != argc) {
        std::fprintf(stderr, "Unexpected extra argument '%s'\n",
                     argv[optind + 1]);
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
}

void output(const CSRGraphTy& graph, const char* output_file) {
    if (output_file == nullptr) {
        return;
    }

    FILE* file = stdout;
    if (std::strcmp(output_file, "-") != 0) {
        file = std::fopen(output_file, "w");
        if (file == nullptr) {
            std::fprintf(stderr, "Unable to open output file %s: %s\n",
                         output_file, std::strerror(errno));
            std::exit(EXIT_FAILURE);
        }
    }

    for (unsigned node = 0; node < graph.nnodes; ++node) {
        if (graph.node_data[node] == INF) {
            std::fprintf(file, "%u INF\n", node);
        } else {
            std::fprintf(file, "%u %.9g\n", node, graph.node_data[node]);
        }
    }

    if (file != stdout) {
        std::fclose(file);
    }
}
