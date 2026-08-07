/* Tile a prepared HIGGS training CSV up to a target row count, streaming to
 * disk without holding the whole thing in RAM.
 *
 * The HIGGS contract (docs/benchmarks/throughput/higgs/README.md) prepares a
 * headerless, comma-separated training file:
 *
 *     feature_0,...,feature_27,label      (28 features + 1 label per row)
 *
 * That file has a fixed row count (up to 10,500,000 for the full split). To
 * probe a RAM budget larger than the file we repeat its rows modulo: output
 * row i is source row (i % source_rows). This lets the capacity sweep push the
 * sample count as high as it needs while every value is a real HIGGS row, not
 * synthetic noise. The tiler's own memory stays constant regardless of the
 * output size — only the loaders under test are meant to run out of memory.
 *
 *   usage:  tile_higgs <higgs_train.csv> <target_rows> <out.csv>
 *
 * Headerless CSV in, headerless CSV out; the row layout is passed through
 * verbatim, so the input feature count (28) is preserved automatically.
 *
 * Portable C: uses only fgets/fputs so it builds with MSVC (cl) alongside the
 * rest of this Windows-CPU benchmark.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#define LINE_MAX_BYTES 4096





int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <higgs_train.csv> <target_rows> <out.csv>\n", argv[0]);
        return 2;
    }

    const char* in_path = argv[1];
    const long long target_rows = atoll(argv[2]);
    const char* out_path = argv[3];

    if (target_rows < 1)
    {
        fprintf(stderr, "need target_rows >= 1\n");
        return 2;
    }

    FILE* in = fopen(in_path, "rb");
    if (!in) { perror("fopen(input)"); return 1; }




    size_t cap = 1 << 20;
    size_t rows = 0;
    char** lines = (char**)malloc(cap * sizeof(char*));
    if (!lines) { fprintf(stderr, "oom allocating line table\n"); return 1; }

    char line[LINE_MAX_BYTES];
    while (fgets(line, (int)sizeof line, in))
    {
        size_t len = strlen(line);
        if (len == 0) continue;



        if (line[len - 1] != '\n' && !feof(in))
        {
            fprintf(stderr, "line %zu exceeds %d bytes; raise LINE_MAX_BYTES\n",
                    rows + 1, LINE_MAX_BYTES);
            return 1;
        }
        if (rows == cap)
        {
            cap *= 2;
            char** grown = (char**)realloc(lines, cap * sizeof(char*));
            if (!grown) { fprintf(stderr, "oom growing line table\n"); return 1; }
            lines = grown;
        }
        char* copy = (char*)malloc(len + 1);
        if (!copy) { fprintf(stderr, "oom copying a source line\n"); return 1; }
        memcpy(copy, line, len + 1);
        lines[rows++] = copy;
    }
    fclose(in);

    if (rows == 0) { fprintf(stderr, "input has no rows: %s\n", in_path); return 1; }



    char* last = lines[rows - 1];
    size_t last_len = strlen(last);
    int last_needs_nl = (last_len == 0 || last[last_len - 1] != '\n');

    FILE* out = fopen(out_path, "wb");
    if (!out) { perror("fopen(output)"); return 1; }

    static char iobuf[1 << 22];
    setvbuf(out, iobuf, _IOFBF, sizeof iobuf);

    for (long long i = 0; i < target_rows; i++)
    {
        const size_t src = (size_t)(i % (long long)rows);
        if (fputs(lines[src], out) == EOF) { perror("fputs"); return 1; }
        if (src == rows - 1 && last_needs_nl) fputc('\n', out);
    }

    for (size_t r = 0; r < rows; r++) free(lines[r]);
    free(lines);

    if (fclose(out) != 0) { perror("fclose"); return 1; }
    return 0;
}
