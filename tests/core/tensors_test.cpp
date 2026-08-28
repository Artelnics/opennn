#include "tests/pch.h"

#include <utility>

#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/statistics.h"
using namespace opennn;

TEST(Tensors, Fill)
{
    MatrixR submatrix;

    vector<Index> rows_indices;
    vector<Index> columns_indices;

    MatrixR matrix(1, 1);
    matrix.setConstant(type(3.1416));

    rows_indices.resize(1, 0);

    columns_indices.resize(1, 0);

    submatrix.resize(1, 1);

    fill_tensor_data(matrix, rows_indices, columns_indices, span<float>(submatrix.data(), size_t(submatrix.size())));

    EXPECT_LT((submatrix.array() - type(3.1416)).abs().maxCoeff(), type(1e-6));
}

TEST(Tensors, HostBfloat16ConversionUsesRoundToNearestEven)
{
    EXPECT_EQ(float_to_bfloat16_host(bit_cast<float>(0x3F808000u)), 0x3F80u);
    EXPECT_EQ(float_to_bfloat16_host(bit_cast<float>(0x3F818000u)), 0x3F82u);
    EXPECT_EQ(float_to_bfloat16_host(bit_cast<float>(0x3F808001u)), 0x3F81u);

    const uint16_t nan = float_to_bfloat16_host(bit_cast<float>(0x7F800001u));
    EXPECT_EQ(nan & 0x7F80u, 0x7F80u);
    EXPECT_NE(nan & 0x007Fu, 0u);
}

TEST(Tensors, IsContiguous)
{
    EXPECT_TRUE(is_contiguous(vector<Index>{ 0, 1, 2, 3 }));
    EXPECT_TRUE(is_contiguous(vector<Index>{ 5, 6 }));
    EXPECT_TRUE(is_contiguous(vector<Index>{ 2 }));

    EXPECT_FALSE(is_contiguous(vector<Index>{ 0, 2 }));
    EXPECT_FALSE(is_contiguous(vector<Index>{ 0, 1, 3 }));
    EXPECT_FALSE(is_contiguous(vector<Index>{ 3, 2, 1 }));
}

TEST(Tensors, FillContiguousColumns)
{
    MatrixR matrix(3, 4);
    for (Index r = 0; r < 3; ++r)
        for (Index c = 0; c < 4; ++c)
            matrix(r, c) = type(r * 10 + c);

    const vector<Index> rows = { 0, 1, 2 };
    const vector<Index> columns = { 1, 2 };

    ASSERT_TRUE(is_contiguous(columns));

    MatrixR submatrix(3, 2);
    fill_tensor_data(matrix, rows, columns, span<float>(submatrix.data(), size_t(submatrix.size())));

    EXPECT_NEAR(submatrix(0, 0), type(1),  1e-6);
    EXPECT_NEAR(submatrix(0, 1), type(2),  1e-6);
    EXPECT_NEAR(submatrix(1, 0), type(11), 1e-6);
    EXPECT_NEAR(submatrix(1, 1), type(12), 1e-6);
    EXPECT_NEAR(submatrix(2, 0), type(21), 1e-6);
    EXPECT_NEAR(submatrix(2, 1), type(22), 1e-6);
}

TEST(Tensors, FillNonContiguousColumns)
{
    MatrixR matrix(3, 4);
    for (Index r = 0; r < 3; ++r)
        for (Index c = 0; c < 4; ++c)
            matrix(r, c) = type(r * 10 + c);

    const vector<Index> rows = { 0, 1, 2 };
    const vector<Index> columns = { 0, 2 };

    ASSERT_FALSE(is_contiguous(columns));

    MatrixR submatrix(3, 2);
    fill_tensor_data(matrix, rows, columns, span<float>(submatrix.data(), size_t(submatrix.size())));

    EXPECT_NEAR(submatrix(0, 0), type(0),  1e-6);
    EXPECT_NEAR(submatrix(0, 1), type(2),  1e-6);
    EXPECT_NEAR(submatrix(1, 0), type(10), 1e-6);
    EXPECT_NEAR(submatrix(1, 1), type(12), 1e-6);
    EXPECT_NEAR(submatrix(2, 0), type(20), 1e-6);
    EXPECT_NEAR(submatrix(2, 1), type(22), 1e-6);
}

TEST(Tensors, FillReordersRowsAndColumns)
{
    MatrixR matrix(3, 4);
    for (Index r = 0; r < 3; ++r)
        for (Index c = 0; c < 4; ++c)
            matrix(r, c) = type(r * 10 + c);

    const vector<Index> rows = { 2, 0 };
    const vector<Index> columns = { 3, 1 };

    ASSERT_FALSE(is_contiguous(columns));

    MatrixR submatrix(2, 2);
    fill_tensor_data(matrix, rows, columns, span<float>(submatrix.data(), size_t(submatrix.size())));

    EXPECT_NEAR(submatrix(0, 0), type(23), 1e-6);
    EXPECT_NEAR(submatrix(0, 1), type(21), 1e-6);
    EXPECT_NEAR(submatrix(1, 0), type(3),  1e-6);
    EXPECT_NEAR(submatrix(1, 1), type(1),  1e-6);
}

TEST(Tensors, FillContiguousHintSelectsPath)
{
    MatrixR matrix(3, 4);
    for (Index r = 0; r < 3; ++r)
        for (Index c = 0; c < 4; ++c)
            matrix(r, c) = type(r * 10 + c);

    const vector<Index> rows = { 0, 1, 2 };
    const vector<Index> columns = { 1, 2 };

    MatrixR memcpy_path(3, 2);
    fill_tensor_data(matrix, rows, columns, span<float>(memcpy_path.data(), size_t(memcpy_path.size())),
                     ColumnContiguity::Contiguous);

    MatrixR gather_path(3, 2);
    fill_tensor_data(matrix, rows, columns, span<float>(gather_path.data(), size_t(gather_path.size())),
                     ColumnContiguity::NonContiguous);

    EXPECT_LT((memcpy_path - gather_path).array().abs().maxCoeff(), type(1e-6));

    EXPECT_NEAR(memcpy_path(0, 0), type(1),  1e-6);
    EXPECT_NEAR(memcpy_path(2, 1), type(22), 1e-6);
}

TEST(Tensors, FillThrowsOnUndersizedOutput)
{
    MatrixR matrix(3, 4);
    matrix.setZero();

    const vector<Index> rows = { 0, 1, 2 };
    const vector<Index> columns = { 1, 2 };

    vector<float> too_small(5);

    EXPECT_THROW(fill_tensor_data(matrix, rows, columns, too_small), runtime_error);

    vector<float> exact(6);

    EXPECT_NO_THROW(fill_tensor_data(matrix, rows, columns, exact));
}

TEST(Shape, DefaultIsEmpty)
{
    Shape shape;

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.get_rank(), 0u);
    EXPECT_EQ(shape.size(), 0);
}

TEST(Shape, InitializerListConstructor)
{
    Shape shape{ 2, 3, 4 };

    EXPECT_FALSE(shape.empty());
    EXPECT_EQ(shape.get_rank(), 3u);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.back(), 4);
    EXPECT_EQ(shape.size(), 24);
}

TEST(Shape, Equality)
{
    EXPECT_EQ((Shape{ 2, 3 }), (Shape{ 2, 3 }));
    EXPECT_NE((Shape{ 2, 3 }), (Shape{ 2, 4 }));
    EXPECT_NE((Shape{ 2, 3 }), (Shape{ 2, 3, 1 }));
}

TEST(Shape, PushBackRejectsRankOverflow)
{
    Shape shape;

    shape.push_back(1);
    shape.push_back(2);

    EXPECT_EQ(shape.get_rank(), 2u);
    EXPECT_EQ(shape.back(), 2);

    shape.push_back(3);
    shape.push_back(4);

    EXPECT_EQ(shape.get_rank(), Shape::MaxRank);
    EXPECT_EQ(shape[3], 4);
    EXPECT_THROW(shape.push_back(5), runtime_error);
}

TEST(Shape, ClearResetsRank)
{
    Shape shape{ 2, 3, 4 };

    shape.clear();

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.size(), 0);
}

TEST(Shape, AppendRejectsRankOverflow)
{
    Shape shape{ 1, 2, 3 };

    EXPECT_THROW(shape.append(Shape{ 4, 5 }), runtime_error);

    EXPECT_EQ(shape, (Shape{1, 2, 3}));
}

TEST(Shape, RejectsNegativeDimensions)
{
    EXPECT_THROW((Shape{2, -1}), runtime_error);

    const vector<Index> dimensions{2, -1};
    EXPECT_THROW((Shape(dimensions.begin(), dimensions.end())), runtime_error);

    Shape shape{2};
    EXPECT_THROW(shape.push_back(-1), runtime_error);
}

TEST(Shape, SetDimensionPreservesInvariants)
{
    static_assert(!is_assignable_v<decltype(declval<Shape&>()[0]), Index>);
    static_assert(!is_assignable_v<decltype(declval<Shape&>().back()), Index>);

    Shape shape{2, 3};
    shape.set_dimension(1, 4);

    EXPECT_EQ(shape, (Shape{2, 4}));
    EXPECT_THROW(shape.set_dimension(2, 5), runtime_error);
    EXPECT_THROW(shape.set_dimension(0, -1), runtime_error);
    EXPECT_EQ(shape, (Shape{2, 4}));
}

TEST(Shape, SizeRejectsIndexOverflow)
{
    const Index maximum = numeric_limits<Index>::max();

    EXPECT_EQ((Shape{maximum, 1}).size(), maximum);
    EXPECT_THROW((Shape{maximum, 2}).size(), runtime_error);
}

TEST(Shape, AlignmentArithmeticRejectsInvalidOrOverflowingValues)
{
    const Index maximum = numeric_limits<Index>::max();

    EXPECT_THROW(align_up(-1, ALIGN_BYTES), runtime_error);
    EXPECT_THROW(align_up(1, 3), runtime_error);
    EXPECT_THROW(align_up(maximum, ALIGN_BYTES), runtime_error);
    EXPECT_THROW(get_aligned_bytes(maximum, Type::FP32), runtime_error);
    EXPECT_EQ(ceil_div(maximum, 2), maximum / 2 + 1);
    EXPECT_THROW(ceil_div(-1, 2), runtime_error);
    EXPECT_THROW(ceil_div(1, 0), runtime_error);
}

TEST(Shape, AggregateAllocationSizeRejectsOverflow)
{
    const Index large_count = numeric_limits<Index>::max() / 8;
    const vector<TensorSpec> specs = {
        {Shape{large_count}, Type::FP32},
        {Shape{large_count}, Type::FP32}
    };

    EXPECT_THROW(get_aligned_bytes(specs), runtime_error);
}

TEST(Shape, DimOrZero)
{
    Shape shape{ 7, 8 };

    EXPECT_EQ(shape.dim_or_zero(0), 7);
    EXPECT_EQ(shape.dim_or_zero(1), 8);
    EXPECT_EQ(shape.dim_or_zero(2), 0);
}

TEST(Shape, BackThrowsOnEmpty)
{
    Shape shape;

    EXPECT_THROW(shape.back(), runtime_error);
}

TEST(Buffer, DefaultIsEmpty)
{
    Buffer buffer;

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.byte_size(), 0);
    EXPECT_EQ(buffer.size_in_floats(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
}

TEST(Buffer, RequiresConcreteDevice)
{
    EXPECT_THROW((void)Buffer{Device::Auto}, runtime_error);

    Buffer buffer;
    EXPECT_THROW(buffer.resize_bytes(0, Device::Auto), runtime_error);
    EXPECT_THROW(buffer.set_view(nullptr, 0, Device::Auto), runtime_error);
    EXPECT_THROW(buffer.migrate_to(Device::Auto), runtime_error);

    EXPECT_EQ(buffer.get_device(), Device::CPU);
    EXPECT_TRUE(buffer.empty());
}

TEST(Buffer, ResizeBytesAllocatesAligned)
{
    Buffer buffer;

    buffer.resize_bytes(16, Device::CPU);

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.byte_size(), 16);
    EXPECT_EQ(buffer.size_in_floats(), 4);
    ASSERT_NE(buffer.data(), nullptr);
    EXPECT_TRUE(is_aligned(buffer.data()));
}

TEST(Buffer, ResizeRejectsInvalidSizeAndPreservesAllocationOnFailure)
{
    Buffer buffer;
    buffer.resize_bytes(Index(sizeof(float)), Device::CPU);
    buffer.as<float>()[0] = type(42);

    void* const original_data = buffer.data();
    EXPECT_THROW(buffer.resize_bytes(-1, Device::CPU), runtime_error);
    EXPECT_THROW(buffer.resize_bytes(numeric_limits<Index>::max(), Device::CPU), exception);

    EXPECT_EQ(buffer.data(), original_data);
    EXPECT_EQ(buffer.byte_size(), Index(sizeof(float)));
    EXPECT_NEAR(buffer.as<float>()[0], type(42), 1e-6);
}

TEST(Buffer, SetViewValidatesMetadataAndReleasesNoExternalStorage)
{
    std::array<float, 2> external{type(3), type(4)};
    Buffer buffer;

    EXPECT_THROW(buffer.set_view(nullptr, Index(sizeof(float)), Device::CPU), runtime_error);
    EXPECT_THROW(buffer.set_view(external.data(), 0, Device::CPU), runtime_error);

    buffer.set_view(external.data(), Index(sizeof(external)), Device::CPU);
    EXPECT_FALSE(buffer.owns_memory());
    EXPECT_EQ(buffer.as<float>(), external.data());

    buffer.resize_bytes(Index(sizeof(external)), Device::CPU);
    EXPECT_TRUE(buffer.owns_memory());
    EXPECT_NE(buffer.data(), external.data());
    EXPECT_EQ(external[0], type(3));
    EXPECT_EQ(external[1], type(4));

    void* const owned_data = buffer.data();
    EXPECT_THROW(buffer.set_view(owned_data, buffer.byte_size(), Device::CPU), runtime_error);
    EXPECT_EQ(buffer.data(), owned_data);
    EXPECT_TRUE(buffer.owns_memory());
}

TEST(Buffer, SetZero)
{
    Buffer buffer;
    buffer.resize_bytes(4 * Index(sizeof(float)), Device::CPU);

    float* data = buffer.as<float>();
    for (Index i = 0; i < 4; ++i)
        data[i] = type(i + 1);

    buffer.setZero();

    for (Index i = 0; i < 4; ++i)
        EXPECT_EQ(data[i], type(0));
}

TEST(Buffer, EnsureReturnsTypedPointer)
{
    Buffer buffer;

    float* data = buffer.ensure<float>(4);

    ASSERT_NE(data, nullptr);
    EXPECT_GE(buffer.byte_size(), 4 * Index(sizeof(float)));

    data[0] = type(7);
    EXPECT_NEAR(buffer.as<float>()[0], type(7), 1e-6);
}

TEST(Buffer, EnsureRejectsByteSizeOverflow)
{
    Buffer buffer;

    EXPECT_THROW(buffer.ensure<float>(numeric_limits<Index>::max()), runtime_error);
    EXPECT_TRUE(buffer.empty());
}

TEST(Buffer, GrowToOnlyGrows)
{
    Buffer buffer;
    buffer.resize_bytes(32, Device::CPU);
    EXPECT_EQ(buffer.byte_size(), 32);

    buffer.grow_to(16);
    EXPECT_EQ(buffer.byte_size(), 32);

    buffer.grow_to(64);
    EXPECT_EQ(buffer.byte_size(), 64);

    EXPECT_THROW(buffer.grow_to(-1), runtime_error);
}

TEST(Buffer, EmptyMigrationChangesFutureAllocationDevice)
{
    Buffer buffer;

    buffer.migrate_to(Device::CUDA);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.get_device(), Device::CUDA);
    EXPECT_TRUE(buffer.owns_memory());
}

TEST(Buffer, VectorAccessRequiresCpuFloatStorage)
{
    Buffer cuda_buffer(Device::CUDA);
    EXPECT_THROW(cuda_buffer.as_vector(), runtime_error);

    Buffer byte_buffer;
    byte_buffer.resize_bytes(3, Device::CPU);
    EXPECT_THROW(byte_buffer.as_vector(), runtime_error);
}

TEST(Buffer, ResizeToZeroFrees)
{
    Buffer buffer;
    buffer.resize_bytes(16, Device::CPU);
    ASSERT_FALSE(buffer.empty());

    buffer.resize_bytes(0, Device::CPU);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
}

TEST(Buffer, MoveTransfersOwnership)
{
    Buffer source;
    source.resize_bytes(8 * Index(sizeof(float)), Device::CPU);
    source.as<float>()[0] = type(42);

    Buffer dest(std::move(source));

    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data(), nullptr);

    EXPECT_FALSE(dest.empty());
    EXPECT_EQ(dest.size_in_floats(), 8);
    EXPECT_NEAR(dest.as<float>()[0], type(42), 1e-6);
}

TEST(TensorView, DefaultIsEmpty)
{
    TensorView view;

    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.get_rank(), 0);
    EXPECT_EQ(view.size(), 0);
    EXPECT_FALSE(view.is_cuda());
}

TEST(TensorView, ExposesReadOnlyMetadata)
{
    static_assert(is_same_v<decltype(declval<const TensorView&>().get_shape()),
                            const Shape&>);
    static_assert(!is_assignable_v<decltype(declval<TensorView&>().get_shape()), Shape>);

    float value = 0.0f;
    const TensorView view(&value, Shape{1}, Type::BF16, Device::CUDA);

    EXPECT_EQ(view.get_data(), &value);
    EXPECT_EQ(view.get_shape(), Shape{1});
    EXPECT_EQ(view.get_type(), Type::BF16);
    EXPECT_EQ(view.get_device(), Device::CUDA);
}

TEST(TensorView, AsMatrixMapsRowMajor)
{
    Tensor2 storage(2, 3);
    storage.setValues({ {type(1), type(2), type(3)},
                        {type(4), type(5), type(6)} });

    TensorView view(storage.data(), { 2, 3 });

    EXPECT_EQ(view.get_rank(), 2);
    EXPECT_EQ(view.size(), 6);
    EXPECT_FALSE(view.empty());

    MatrixMap matrix = view.as_matrix();

    ASSERT_EQ(matrix.rows(), 2);
    ASSERT_EQ(matrix.cols(), 3);
    EXPECT_NEAR(matrix(0, 0), type(1), 1e-6);
    EXPECT_NEAR(matrix(0, 2), type(3), 1e-6);
    EXPECT_NEAR(matrix(1, 0), type(4), 1e-6);
    EXPECT_NEAR(matrix(1, 2), type(6), 1e-6);
}

TEST(TensorView, AsFlatMatrixCollapsesLeadingDimensions)
{
    Tensor3 storage(2, 3, 4);
    storage.setZero();

    TensorView view(storage.data(), { 2, 3, 4 });

    MatrixMap flat = view.as_flat_matrix();

    EXPECT_EQ(flat.rows(), 6);
    EXPECT_EQ(flat.cols(), 4);
}

TEST(TensorView, AsVectorFlattens)
{
    Tensor2 storage(2, 2);
    storage.setValues({ {type(1), type(2)},
                        {type(3), type(4)} });

    TensorView view(storage.data(), { 2, 2 });

    VectorMap vector = view.as_vector();

    ASSERT_EQ(vector.size(), 4);
    EXPECT_NEAR(vector(0), type(1), 1e-6);
    EXPECT_NEAR(vector(3), type(4), 1e-6);
}

TEST(TensorView, EigenMapsRequireCpuFp32Storage)
{
    std::array<float, 4> storage{};
    const Shape shape{2, 2};

    TensorView cuda_view(storage.data(), shape, Type::FP32, Device::CUDA);
    EXPECT_THROW(cuda_view.as_matrix(), runtime_error);
    EXPECT_THROW(cuda_view.as_flat_matrix(), runtime_error);
    EXPECT_THROW(cuda_view.as_vector(), runtime_error);

    TensorView bf16_view(storage.data(), shape, Type::BF16, Device::CPU);
    EXPECT_THROW(bf16_view.as_matrix(0), runtime_error);
    EXPECT_THROW(bf16_view.as_tensor<2>(), runtime_error);
    EXPECT_THROW(bf16_view.as_tensor<1>(0), runtime_error);
}

TEST(TensorView, SlotOrUsesCallerOwnedFallback)
{
    float value = 1.0f;
    vector<TensorView> views{TensorView(&value, Shape{1})};
    const vector<size_t> present_slot{0};
    const vector<size_t> missing_slot;
    TensorView fallback;

    EXPECT_EQ(&slot_or(views, present_slot, 0, fallback), &views[0]);
    EXPECT_EQ(&slot_or(views, missing_slot, 0, fallback), &fallback);
}

TEST(TensorView, SlotOrRejectsInvalidMappedIndex)
{
    vector<TensorView> views(1);
    const vector<size_t> invalid_slot{1};
    TensorView fallback;

    EXPECT_THROW(slot_or(views, invalid_slot, 0, fallback), runtime_error);
}

TEST(TensorOperationsValidation, CopyRequiresMatchingMetadata)
{
    std::array<float, 4> source_storage{};
    std::array<float, 4> destination_storage{};
    TensorView source(source_storage.data(), {2, 2});

    TensorView reshaped(destination_storage.data(), {1, 4});
    EXPECT_THROW(copy(source, reshaped), runtime_error);

    TensorView cuda_destination(destination_storage.data(), {2, 2}, Type::FP32, Device::CUDA);
    EXPECT_THROW(copy(source, cuda_destination), runtime_error);

    TensorView bf16_destination(destination_storage.data(), {2, 2}, Type::BF16, Device::CPU);
    EXPECT_THROW(copy(source, bf16_destination), runtime_error);
}

TEST(TensorOperationsValidation, AddRequiresMatchingMetadata)
{
    std::array<float, 4> storage{};
    TensorView input_1(storage.data(), {2, 2});
    TensorView input_2(storage.data(), {2, 2});
    TensorView wrong_shape(storage.data(), {1, 4});
    TensorView wrong_type(storage.data(), {2, 2}, Type::BF16, Device::CPU);

    EXPECT_THROW(add(input_1, input_2, wrong_shape), runtime_error);
    EXPECT_THROW(add(input_1, input_2, wrong_type), runtime_error);
}

TEST(TensorOperationsValidation, MultiplyValidatesMatrixAndOutputShapes)
{
    std::array<float, 12> input_a_storage{};
    std::array<float, 20> input_b_storage{};
    std::array<float, 8> output_storage{};
    TensorView input_a(input_a_storage.data(), {2, 3});
    TensorView wrong_inner(input_b_storage.data(), {4, 5});
    TensorView output(output_storage.data(), {2, 4});

    EXPECT_THROW(multiply(input_a, opennn::Transpose::No, wrong_inner, opennn::Transpose::No, output), runtime_error);

    TensorView input_b(input_b_storage.data(), {3, 4});
    TensorView wrong_output(output_storage.data(), {1, 8});
    EXPECT_THROW(multiply(input_a, opennn::Transpose::No, input_b, opennn::Transpose::No, wrong_output), runtime_error);
}

TEST(TensorOperationsValidation, ActivationBackwardRequiresMatchingTensors)
{
    std::array<float, 4> storage{};
    TensorView outputs(storage.data(), {2, 2});
    TensorView wrong_delta(storage.data(), {1, 4});

    EXPECT_THROW(activation_backward(outputs, wrong_delta, ActivationFunction::ReLU), runtime_error);
}

TEST(TensorOperationsValidation, LinearForwardValidatesOutputAndBiasShapes)
{
    std::array<float, 6> input_storage{};
    std::array<float, 12> weight_storage{};
    std::array<float, 8> output_storage{};
    std::array<float, 4> bias_storage{};
    TensorView input(input_storage.data(), {2, 3});
    TensorView weights(weight_storage.data(), {3, 4});
    TensorView output(output_storage.data(), {2, 4});
    TensorView wrong_output(output_storage.data(), {1, 8});
    TensorView wrong_bias(bias_storage.data(), {2, 2});
    TensorView empty;

    EXPECT_THROW(linear_forward(input, weights, empty, wrong_output), runtime_error);
    EXPECT_THROW(linear_forward(input, weights, wrong_bias, output), runtime_error);
}

TEST(TensorOperationsValidation, LinearBackwardValidatesGradientShapes)
{
    std::array<float, 6> input_storage{};
    std::array<float, 12> weight_storage{};
    std::array<float, 8> output_delta_storage{};
    std::array<float, 12> gradient_storage{};
    TensorView input(input_storage.data(), {2, 3});
    TensorView weights(weight_storage.data(), {3, 4});
    TensorView output_delta(output_delta_storage.data(), {2, 4});
    TensorView wrong_weight_gradient(gradient_storage.data(), {4, 3});
    TensorView empty;

    EXPECT_THROW(linear_backward(output_delta, input, weights,
                                 wrong_weight_gradient, empty, empty), runtime_error);
}

TEST(TensorView, ReshapePreservesDataPointer)
{
    Tensor2 storage(2, 3);
    storage.setZero();

    TensorView view(storage.data(), { 2, 3 });
    TensorView reshaped = view.reshape({ 3, 2 });

    EXPECT_EQ(reshaped.get_rank(), 2);
    EXPECT_EQ(reshaped.get_shape()[0], 3);
    EXPECT_EQ(reshaped.get_shape()[1], 2);
    EXPECT_EQ(reshaped.size(), 6);
    EXPECT_EQ(reshaped.as<type>(), view.as<type>());

    EXPECT_THROW(view.reshape({2, 2}), runtime_error);

    TensorView prefix = view.reshape_prefix({2, 2});
    EXPECT_EQ(prefix.get_shape(), (Shape{2, 2}));
    EXPECT_EQ(prefix.as<type>(), view.as<type>());
    EXPECT_THROW(view.reshape_prefix({7}), runtime_error);
}

TEST(TensorView, ByteSizeRejectsIndexOverflow)
{
    float value = 0.0f;
    TensorView view(&value, Shape{numeric_limits<Index>::max()});

    EXPECT_THROW(view.byte_size(), runtime_error);
}

TEST(TensorView, WriteThroughViewModifiesBuffer)
{
    Tensor2 storage(2, 2);
    storage.setZero();

    TensorView view(storage.data(), { 2, 2 });

    view.as<type>()[0] = type(9);

    MatrixMap matrix = view.as_matrix();
    matrix(1, 1) = type(7);

    EXPECT_NEAR(storage(0, 0), type(9), 1e-6);
    EXPECT_NEAR(storage(1, 1), type(7), 1e-6);
}

TEST(Type, TypeBytes)
{
    EXPECT_EQ(type_bytes(Type::FP32), 4);

    EXPECT_GT(type_bytes(Type::BF16), 0);
    EXPECT_LT(type_bytes(Type::BF16), type_bytes(Type::FP32));

    EXPECT_THROW(type_bytes(Type::Auto), runtime_error);
}

TEST(Shape, ToStringRoundTrip)
{
    const Shape original{ 2, 3, 4 };
    EXPECT_EQ(string_to_shape(shape_to_string(original)), original);

    const Shape single{ 5 };
    EXPECT_EQ(string_to_shape(shape_to_string(single)), single);
}

TEST(Shape, ToStringCustomSeparator)
{
    EXPECT_EQ(shape_to_string(Shape{ 2, 3 }, "x"), "2x3x");
    EXPECT_EQ(string_to_shape("2x3", "x"), (Shape{ 2, 3 }));
}

TEST(Shape, ToStringEmptyThrows)
{
    EXPECT_THROW(shape_to_string(Shape{}), runtime_error);
}
