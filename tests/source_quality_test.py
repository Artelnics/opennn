"""Source inventory, CUDA metrics and dependency-boundary regression checks."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_architecture as architecture
import check_code_quality as quality


SPDX = "// SPDX-License-Identifier: LGPL-2.1-or-later\n"
HAS_LIZARD = importlib.util.find_spec("lizard") is not None


class SourceFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "opennn"

    def write(self, name, code, standard_header=True):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((SPDX if standard_header else "") + code, encoding="utf-8")
        return path


class SourceArchitectureTest(SourceFixture):
    def test_inventory_includes_cuda_and_only_excludes_the_vendor_directory(self):
        expected = {
            self.write("core/function.cpp", "void f() {}"),
            self.write("core/function.h", "void f();"),
            self.write("core/cuda/function.cu", "__global__ void f() {}"),
            self.write("core/cuda/function.cuh", "__device__ void f();"),
            self.write("network/flash_attention_shim/first_party.cu", "void f() {}"),
        }
        self.write("core/cuda/flash_attention_shim/vendor.cuh", "vendor code", False)
        self.write("core/notes.txt", "notes", False)
        self.assertEqual(set(quality.source_files(self.source)), expected)

    def test_forbidden_dependencies_are_detected_in_both_cuda_extensions(self):
        for suffix in (".cu", ".cuh"):
            self.write(f"core/cuda/invalid{suffix}",
                       '# include "opennn/training/optimizer.h"\n')
        self.write("core/cuda/flash_attention_shim/vendor.cuh",
                   '#include "opennn/network/network.h"\n', False)
        failures = architecture.check(self.source)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("core must not depend on training" in item for item in failures))
        self.assertTrue(any("invalid.cu:2:" in item for item in failures))
        self.assertTrue(any("invalid.cuh:2:" in item for item in failures))

    def test_cuda_implementation_types_do_not_relax_portable_header_boundary(self):
        self.write("network/kernel.cuh", "void launch(cudaStream_t stream);\n")
        self.write("network/operator.h", "void launch(cudaStream_t stream);\n")
        failures = architecture.check(self.source)
        self.assertEqual(len(failures), 1)
        self.assertIn("operator.h:2: expose an OpenNN backend type", failures[0])

    def test_historical_exceptions_do_not_extend_to_cuda_siblings(self):
        self.write("dataset/bert_dataset.cpp", '#include "opennn/network/network.h"\n')
        self.write("dataset/bert_dataset.cu", '#include "opennn/network/network.h"\n')
        failures = architecture.check(self.source)
        self.assertEqual(len(failures), 1)
        self.assertIn("bert_dataset.cu:2: dataset must not depend on network", failures[0])

    def test_every_metric_requires_a_limit_and_cpp_limits_remain_independent(self):
        metrics = {"functions": 3, "cuda_functions": 1, "combined_duplicate_percent": 2.0}
        failures = quality.limit_failures(metrics, {"functions": 2, "cuda_functions": 100})
        self.assertIn("functions: 3 exceeds reviewed limit 2", failures)
        self.assertIn("combined_duplicate_percent: missing reviewed limit", failures)
        self.assertEqual(quality.limit_failures(metrics, metrics), [])


@unittest.skipUnless(HAS_LIZARD, "requires tools/code-quality-requirements.txt")
class SourceMetricsTest(SourceFixture):
    def test_cuda_templates_kernels_branches_and_launches_are_measured(self):
        self.write("core/cuda/kernel.cu", """template<class T>
__global__ void clamp(T* data, int n) {
    const int i = threadIdx.x;
    if (i < n) {
        if (data[i] < 0) data[i] = 0;
        else if (data[i] > 1) data[i] = 1;
    }
}
__device__ int sign(int n) { return n < 0 ? -1 : 1; }
void launch(float* data, int n) { clamp<<<1, 32>>>(data, n); }
""")
        self.write("core/cuda/helper.cuh",
                   "__host__ __device__ inline int zero() { return 0; }\n")
        metrics = quality.analyze(self.source)
        self.assertEqual(metrics["files"], 0)
        self.assertEqual(metrics["cuda_files"], 2)
        self.assertEqual(metrics["cuda_functions"], 4)
        self.assertEqual(metrics["cuda_maximum_function_ccn"], 4)
        self.assertGreater(metrics["cuda_nloc"], 0)
        self.assertGreater(metrics["cuda_physical_lines"], metrics["cuda_nloc"])

    def test_duplication_is_measured_within_cuda_and_across_cpp_cuda(self):
        code = """template<class T> T projected(T input, int count) {
    T total = 0;
    for (int i = 0; i < count; ++i) {
        T weight = input * T(i + 1);
        if (weight > total) total = weight;
        else total = total / (weight + T(1));
        if (i % 2 == 0) total += input;
        else total -= input / T(i + 1);
    }
    return total > 0 ? total : -total;
}
"""
        self.write("core/shared.cpp", code)
        self.write("core/cuda/shared.cu", code)
        metrics = quality.analyze(self.source)
        self.assertEqual(metrics["duplicate_percent"], 0)
        self.assertEqual(metrics["cuda_duplicate_percent"], 0)
        self.assertGreater(metrics["combined_duplicate_percent"], 0)
        self.write("core/cuda/copied.cuh", code)
        self.assertGreater(quality.analyze(self.source)["cuda_duplicate_percent"], 0)

    def test_standard_source_header_is_required_in_cuda(self):
        for suffix in (".cu", ".cuh"):
            with self.subTest(suffix=suffix):
                path = self.write(f"core/cuda/kernel{suffix}", "void f() {}", False)
                with self.assertRaisesRegex(SystemExit, "Missing standard source header"):
                    quality.analyze(self.source)
                path.write_text(SPDX + "void f() {}", encoding="utf-8")

    def test_declaration_only_files_have_zero_function_metrics(self):
        self.write("core/api.h", "void f();\n")
        metrics = quality.analyze(self.source)
        self.assertEqual(metrics["files"], 1)
        self.assertEqual(metrics["functions"], 0)
        self.assertEqual(metrics["maximum_function_nloc"], 0)
        self.assertEqual(metrics["maximum_function_ccn"], 0)
        self.assertEqual(metrics["cuda_files"], 0)


if __name__ == "__main__":
    unittest.main()
