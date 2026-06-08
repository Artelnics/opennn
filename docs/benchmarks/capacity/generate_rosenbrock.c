/* Stream a Rosenbrock approximation CSV to disk without holding it in RAM.
 *
 *   x_i ~ U(-1, 1),   y = sum_i [ (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2 ]
 *
 * Matches the formula and input range used by the rest of the OpenNN
 * benchmark suite (docs/benchmarks/accuracy/generate_rosenbrock.py). The file
 * is written row by row so the generator's own memory stays constant
 * regardless of file size — only the loaders under test are meant to run out
 * of memory, not this tool.
 *
 *   usage:  generate_rosenbrock <variables> <samples> <out.csv> [seed]
 *
 * Headerless CSV, comma-separated: <variables> inputs then 1 target per row.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* xorshift128+ — fast, deterministic, good enough for benchmark data. */
static uint64_t s0, s1;
static double next_uniform(void)            /* returns a double in [-1, 1) */
{
    uint64_t x = s0;
    const uint64_t y = s1;
    s0 = y;
    x ^= x << 23;
    s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    const uint64_t r = s1 + y;
    /* top 53 bits -> [0,1), then map to [-1,1) */
    return ((double)(r >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <variables> <samples> <out.csv> [seed]\n", argv[0]);
        return 2;
    }

    const long variables = atol(argv[1]);
    const long long samples = atoll(argv[2]);
    const char* out_path = argv[3];
    const uint64_t seed = (argc > 4) ? strtoull(argv[4], NULL, 10) : 1234ULL;

    if (variables < 2 || samples < 1)
    {
        fprintf(stderr, "need variables >= 2 and samples >= 1\n");
        return 2;
    }

    s0 = seed ? seed : 0x9E3779B97F4A7C15ULL;
    s1 = seed ^ 0xBF58476D1CE4E5B9ULL;
    for (int i = 0; i < 16; i++) next_uniform();   /* warm up */

    FILE* f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); return 1; }

    /* Big stdio buffer to keep throughput high on multi-GB files. */
    static char iobuf[1 << 22];
    setvbuf(f, iobuf, _IOFBF, sizeof iobuf);

    double* x = (double*)malloc((size_t)variables * sizeof(double));
    if (!x) { fprintf(stderr, "oom allocating one row\n"); return 1; }

    for (long long row = 0; row < samples; row++)
    {
        double y = 0.0;
        for (long i = 0; i < variables; i++) x[i] = next_uniform();
        for (long i = 0; i + 1 < variables; i++)
        {
            const double a = 1.0 - x[i];
            const double b = x[i + 1] - x[i] * x[i];
            y += a * a + 100.0 * b * b;
        }

        for (long i = 0; i < variables; i++) fprintf(f, "%.6g,", x[i]);
        fprintf(f, "%.6g\n", y);
    }

    free(x);
    if (fclose(f) != 0) { perror("fclose"); return 1; }
    return 0;
}
