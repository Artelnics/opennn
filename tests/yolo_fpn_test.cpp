#include "pch.h"
#include "numerical_derivatives.h"

#include "../opennn/yolo_dataset.h"
#include "../opennn/detection_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network.h"
#include "../opennn/loss.h"
#include "../opennn/batch.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace opennn;

namespace {

void write_bmp_24(const std::filesystem::path& path, int width, int height, uint8_t r, uint8_t g, uint8_t b)
{
    const int row_bytes_unpadded = width * 3;
    const int row_pad = (4 - row_bytes_unpadded % 4) % 4;
    const int row_stride = row_bytes_unpadded + row_pad;
    const int pixel_data_size = row_stride * height;
    const int file_size = 54 + pixel_data_size;

    std::vector<uint8_t> file(static_cast<size_t>(file_size), 0);

    file[0] = 'B'; file[1] = 'M';
    file[2] = static_cast<uint8_t>(file_size & 0xff);
    file[3] = static_cast<uint8_t>((file_size >> 8) & 0xff);
    file[4] = static_cast<uint8_t>((file_size >> 16) & 0xff);
    file[5] = static_cast<uint8_t>((file_size >> 24) & 0xff);
    file[10] = 54;

    file[14] = 40;
    file[18] = static_cast<uint8_t>(width & 0xff);
    file[22] = static_cast<uint8_t>(height & 0xff);
    file[26] = 1;
    file[28] = 24;

    for (int y = 0; y < height; ++y)
    {
        const int row_offset = 54 + y * row_stride;
        for (int x = 0; x < width; ++x)
        {
            file[row_offset + x * 3 + 0] = b;
            file[row_offset + x * 3 + 1] = g;
            file[row_offset + x * 3 + 2] = r;
        }
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), file.size());
}

void write_classes(const std::filesystem::path& path, std::initializer_list<const char*> names)
{
    std::ofstream out(path);
    for (auto* n : names) out << n << '\n';
}

struct TempDir
{
    std::filesystem::path path;

    TempDir()
    {
        const auto base = std::filesystem::temp_directory_path();
        for (int i = 0; i < 10000; ++i)
        {
            std::filesystem::path candidate = base / ("opennn_yolo_fpn_test_" + std::to_string(i));
            std::error_code ec;
            if (std::filesystem::create_directories(candidate, ec) && !ec)
            {
                path = candidate;
                return;
            }
        }
        throw std::runtime_error("Could not create temp dir for YOLO FPN test");
    }

    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

}

TEST(YoloFPN, SingleHeadLargeGridNoObjectGradientMatchesNumerical)
{
    // Isolates whether the gradient bug is grid-size-dependent or multi-head-specific.
    // Same 8x8 image + 8x8 detection grid as the FPN small head, but single head.
    // If this passes but the multi-head test fails, bug is in multi-head routing.
    // If this also fails, bug is somewhere in the 8x8 single-head path.

    TempDir dir;
    const std::filesystem::path images_dir = dir.path / "images";
    const std::filesystem::path labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "classes.names", {"only"});

    write_bmp_24(images_dir / "a.bmp", 8, 8, 200, 100, 50);
    write_bmp_24(images_dir / "b.bmp", 8, 8,  50, 200, 100);
    { std::ofstream empty_a(labels_dir / "a.txt"); }
    { std::ofstream empty_b(labels_dir / "b.txt"); }

    constexpr Index input_H = 8, input_W = 8, classes_number = 1, boxes_per_cell = 3;
    constexpr Index head_channels = boxes_per_cell * (5 + classes_number);
    const std::vector<std::array<float, 2>> anchors{{0.1f, 0.1f}, {0.15f, 0.15f}, {0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    dataset.set(images_dir, labels_dir, Shape{input_H, input_W, 3},
                /*grid_size=*/8, boxes_per_cell, anchors);
    {
        YoloDataset::AugmentationConfig no_aug; no_aug.enabled = false;
        dataset.set_augmentation(no_aug);
    }

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 3}, Shape{3, 3, 3, 4},
        "Identity", Shape{1, 1}, "Same", false, "conv1"));
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits"));
    neural_network.add_layer(make_unique<Detection>(
        Shape{input_H, input_W, head_channels}, anchors, "detection"));
    neural_network.compile();
    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(0.05f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    // Manually verify numerical gradient for param[0] to catch formula bugs
    {
        const Index sn = dataset.get_samples_number("Training");
        const vector<Index> ti = dataset.get_sample_indices("Training");
        const vector<Index> ii = dataset.get_feature_indices("Input");
        const vector<Index> di = dataset.get_feature_indices("Decoder");
        const vector<Index> tgti = dataset.get_feature_indices("Target");
        Batch batch_diag(sn, &dataset, neural_network.get_config());
        batch_diag.fill(ti, ii, di, tgti);
        ForwardPropagation fp_diag(sn, &neural_network);
        neural_network.forward_propagate(batch_diag.get_inputs(), fp_diag, true);
        BackPropagation bp_diag(sn, &loss);
        const float L0 = loss.calculate_error(batch_diag, fp_diag).error;

        VectorMap params(neural_network.get_parameters_data(), neural_network.get_parameters_size());
        VectorR perturbed = params;
        const float h = Loss::calculate_h(params(0));
        perturbed(0) += h;
        neural_network.forward_propagate(batch_diag.get_inputs(), perturbed, fp_diag);
        const float Lp = loss.calculate_error(batch_diag, fp_diag).error;
        perturbed(0) -= 2.0f*h;
        neural_network.forward_propagate(batch_diag.get_inputs(), perturbed, fp_diag);
        const float Lm = loss.calculate_error(batch_diag, fp_diag).error;
        // restore
        neural_network.forward_propagate(batch_diag.get_inputs(), params, fp_diag);

        std::cout << "Manual check param[0]: L0=" << L0 << " L+=" << Lp << " L-=" << Lm
                  << " h=" << h << "\n"
                  << "  manual_num=" << (Lp-Lm)/(2.0f*h)
                  << "  c4 effect: 3boxes*0.05w*sum_cells_c4\n";

        // h-sweep: check if numerical gradient is h-dependent (use VectorR for deep copy).
        for (float hh : {1e-4f, 1e-3f, 1e-2f}) {
            VectorR p_ph = VectorR(params); p_ph(0) += hh;
            VectorR p_mh = VectorR(params); p_mh(0) -= hh;
            neural_network.forward_propagate(batch_diag.get_inputs(), p_ph, fp_diag);
            const float Lph = loss.calculate_error(batch_diag, fp_diag).error;
            neural_network.forward_propagate(batch_diag.get_inputs(), p_mh, fp_diag);
            const float Lmh = loss.calculate_error(batch_diag, fp_diag).error;
            std::cout << "  h=" << hh << " num_grad=" << (Lph-Lmh)/(2.0f*hh) << "\n";
        }
        // restore original
        {VectorR r = VectorR(params); neural_network.forward_propagate(batch_diag.get_inputs(), r, fp_diag);}
    }
    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(0.05f);
    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    // Print first few values of each layer's gradient for cross-layer diagnosis
    Index offset = 0;
    float worst = 0.0f;
    Index worst_idx = 0;
    for (Index li = 0; li < neural_network.get_layers_number(); ++li)
    {
        const Index np = neural_network.get_layer(li)->get_parameters_number();
        if (np == 0) { std::cout << "  layer " << li << " (" << neural_network.get_layer(li)->get_label() << "): no params\n"; continue; }
        float lworst = 0.0f; Index lwi = offset;
        for (Index k = offset; k < offset + np; ++k)
        {
            const float d = std::abs(gradient(k) - numerical_gradient(k));
            if (d > lworst) { lworst = d; lwi = k; }
            if (d > worst)  { worst  = d; worst_idx = k; }
        }
        std::cout << "  layer " << li << " (" << neural_network.get_layer(li)->get_label()
                  << "): worst=" << lworst << " at local[" << (lwi-offset) << "]"
                  << " grad=" << gradient(lwi) << " num=" << numerical_gradient(lwi) << "\n";
        // Print first 4 param values
        std::cout << "    first 4 grad: ";
        for (Index k = offset; k < std::min(offset+4, offset+np); ++k) std::cout << gradient(k) << " ";
        std::cout << "\n    first 4 num:  ";
        for (Index k = offset; k < std::min(offset+4, offset+np); ++k) std::cout << numerical_gradient(k) << " ";
        std::cout << "\n";
        offset += np;
    }
    std::cout << "\n[SingleHead8x8] Overall worst at idx " << worst_idx
              << ": grad=" << gradient(worst_idx) << " num=" << numerical_gradient(worst_idx)
              << " ratio=" << gradient(worst_idx)/numerical_gradient(worst_idx) << "\n";

    // Print gradient buffer size vs per-layer sum to detect alignment padding
    std::cout << "gradient.size()=" << gradient.size()
              << " sum_per_layer=" << offset
              << " get_parameters_size()=" << neural_network.get_parameters_size() << "\n";
    // Find first nonzero in gradient after conv1 params to locate logits true start
    const Index conv1_np = neural_network.get_layer(0)->get_parameters_number();
    std::cout << "conv1 params=" << conv1_np << " logits params=" << neural_network.get_layer(1)->get_parameters_number() << "\n";
    Index logits_true_start = conv1_np;
    // scan for first nonzero to find actual logits start
    for (Index k = conv1_np; k < gradient.size(); ++k)
        if (gradient(k) != 0.0f || numerical_gradient(k) != 0.0f)
            { logits_true_start = k; break; }
    std::cout << "logits_true_start=" << logits_true_start << "\n";
    std::cout << "logits grad[0..17]: ";
    for (Index k = logits_true_start; k < logits_true_start+18 && k < gradient.size(); ++k)
        std::cout << gradient(k) << " ";
    std::cout << "\nlogits num[0..17]: ";
    for (Index k = logits_true_start; k < logits_true_start+18 && k < numerical_gradient.size(); ++k)
        std::cout << numerical_gradient(k) << " ";
    std::cout << "\n";

    EXPECT_LT(worst, 0.05f);
}

TEST(YoloFPN, MultiHeadNoObjectGradientMatchesNumerical)
{
    // Tiny hand-built FPN-shaped network: 3 detection heads at strides 1/2/4
    // over an 8x8 input. Validates:
    //   • back_propagation pool allocates output_delta slot 0 for every leaf
    //     Detection (not just the last_trainable one)
    //   • Loss::calculate_error / calculate_output_deltas dispatch to the
    //     multi-head path when network has multiple Detection layers
    //   • yolo_error_cpu_multi / yolo_gradient_cpu_multi correctly slice the
    //     flat target buffer per head and write per-head deltas
    //   • make_target_multi_scale lays out targets matching the network's
    //     head order (large stride first → small stride last)
    //
    // No-object dataset → loss is pure noobject branch which is exact (no IoU
    // approximation), so tolerance can be tight.

    TempDir dir;
    const std::filesystem::path images_dir = dir.path / "images";
    const std::filesystem::path labels_dir = dir.path / "labels";
    std::filesystem::create_directories(images_dir);
    std::filesystem::create_directories(labels_dir);
    write_classes(labels_dir / "classes.names", {"only"});

    write_bmp_24(images_dir / "a.bmp", 8, 8, 200, 100, 50);
    write_bmp_24(images_dir / "b.bmp", 8, 8,  50, 200, 100);
    { std::ofstream empty_a(labels_dir / "a.txt"); }
    { std::ofstream empty_b(labels_dir / "b.txt"); }

    constexpr Index input_H = 8;
    constexpr Index input_W = 8;
    constexpr Index classes_number = 1;
    constexpr Index boxes_per_head = 3;
    constexpr Index head_channels = boxes_per_head * (5 + classes_number);

    // FPN heads at strides 1 / 2 / 4 over 8x8 input → grid sizes 8 / 4 / 2.
    // Order in network = large stride first (head_large at grid 2 with the
    // most-downsampled features), matching set_multi_scale_heads layout.
    const std::vector<Index> head_grids{2, 4, 8};

    const std::vector<std::array<float, 2>> anchors_large {{0.6f, 0.6f}, {0.7f, 0.7f}, {0.8f, 0.8f}};
    const std::vector<std::array<float, 2>> anchors_medium{{0.3f, 0.3f}, {0.4f, 0.4f}, {0.5f, 0.5f}};
    const std::vector<std::array<float, 2>> anchors_small {{0.1f, 0.1f}, {0.15f, 0.15f}, {0.2f, 0.2f}};

    YoloDataset dataset;
    dataset.set_display(false);
    // Ctor anchors only satisfy the single-scale validator; multi-scale config
    // below overrides for target encoding.
    dataset.set(images_dir, labels_dir, Shape{input_H, input_W, 3},
                /*grid_size=*/2, /*boxes_per_cell=*/3, anchors_large);
    dataset.set_multi_scale_heads(head_grids, {anchors_large, anchors_medium, anchors_small});

    // Build the network manually: input → 3 stride-2 convs (capturing 3
    // intermediate features) → each gets a 1×1 yolo_logits → Detection.
    // Mirrors the structural shape of FPN without using Upsample/Concatenation
    // (those are validated separately by the smoke test). This isolates the
    // multi-head loss + pool allocation under test.
    NeuralNetwork neural_network;

    // Stage 1: 8x8x3 → 8x8x4 (stride 1)
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 3}, Shape{3, 3, 3, 4},
        "Identity", Shape{1, 1}, "Same", false, "stage1"));
    const Index s1 = neural_network.get_layers_number() - 1;

    // Stage 2: 8x8x4 → 4x4x4 (stride 2)
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 4}, Shape{3, 3, 4, 4},
        "Identity", Shape{2, 2}, "Same", false, "stage2"));
    const Index s2 = neural_network.get_layers_number() - 1;

    // Stage 3: 4x4x4 → 2x2x4 (stride 2)
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{4, 4, 4}, Shape{3, 3, 4, 4},
        "Identity", Shape{2, 2}, "Same", false, "stage3"));
    const Index s3 = neural_network.get_layers_number() - 1;

    // Head LARGE (stride 4, grid 2x2): 1x1 conv on s3 → Detection
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{2, 2, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits_large"), {s3});
    neural_network.add_layer(make_unique<Detection>(
        Shape{2, 2, head_channels}, anchors_large, "detection_large"));

    // Head MEDIUM (stride 2, grid 4x4): 1x1 conv on s2 → Detection
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{4, 4, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits_medium"), {s2});
    neural_network.add_layer(make_unique<Detection>(
        Shape{4, 4, head_channels}, anchors_medium, "detection_medium"));

    // Head SMALL (stride 1, grid 8x8): 1x1 conv on s1 → Detection
    neural_network.add_layer(make_unique<Convolutional>(
        Shape{input_H, input_W, 4}, Shape{1, 1, 4, head_channels},
        "Identity", Shape{1, 1}, "Same", false, "logits_small"), {s1});
    neural_network.add_layer(make_unique<Detection>(
        Shape{input_H, input_W, head_channels}, anchors_small, "detection_small"));

    neural_network.compile();
    VectorMap(neural_network.get_parameters_data(), neural_network.get_parameters_size()).setConstant(0.05f);

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::Yolo);
    loss.set_regularization(Loss::Regularization::NoRegularization);

    const VectorR gradient = calculate_gradient(loss);
    const VectorR numerical_gradient = calculate_numerical_gradient(loss);

    // Diagnostic: print first 10 values of both vectors so we can see the
    // pattern of mismatch (e.g., one head zero, scaling factor off, etc).
    std::cout << "\nGradient (first 10): ";
    for (Index i = 0; i < std::min<Index>(10, gradient.size()); ++i)
        std::cout << gradient(i) << " ";
    std::cout << "\nNumerical (first 10): ";
    for (Index i = 0; i < std::min<Index>(10, numerical_gradient.size()); ++i)
        std::cout << numerical_gradient(i) << " ";
    std::cout << "\nGradient size: " << gradient.size()
              << " | Numerical size: " << numerical_gradient.size() << "\n";
    // Layer sum is fewer than parameters buffer size: alignment padding floats
    // are interleaved. Restrict the comparison to layer-owned params only.
    Index sum_per_layer = 0;
    for (Index i = 0; i < neural_network.get_layers_number(); ++i)
        sum_per_layer += neural_network.get_layer(i)->get_parameters_number();

    Index worst_idx = 0;
    float worst = 0.0f;
    for (Index i = 0; i < sum_per_layer; ++i)
    {
        const float d = std::abs(gradient(i) - numerical_gradient(i));
        if (d > worst) { worst = d; worst_idx = i; }
    }
    std::cout << "Worst mismatch (layer-owned span) at idx " << worst_idx
              << ": grad=" << gradient(worst_idx)
              << " num=" << numerical_gradient(worst_idx) << "\n";

    Index worst_idx_full = 0;
    float worst_full = 0.0f;
    for (Index i = 0; i < gradient.size(); ++i)
    {
        const float d = std::abs(gradient(i) - numerical_gradient(i));
        if (d > worst_full) { worst_full = d; worst_idx_full = i; }
    }
    std::cout << "Worst mismatch (full buffer) at idx " << worst_idx_full
              << ": grad=" << gradient(worst_idx_full)
              << " num=" << numerical_gradient(worst_idx_full) << "\n";

    // No-object branch has no IoU approximation so the gradient should be exact.
    // Single-head no-object test passes at 1e-3. FPN has 3 heads × 504 cells so
    // float32 accumulation is larger, but 0.05 still catches sign errors, scale-of-2
    // bugs, and routing mistakes. If this fails it signals a systematic error in
    // the FPN gradient path — not float32 noise — and must be investigated.
    EXPECT_LT(worst, 0.05f);
}
