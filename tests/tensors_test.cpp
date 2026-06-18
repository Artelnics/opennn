#include "pch.h"

#include "../opennn/tensor_types.h"
#include "../opennn/statistics.h"
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

    fill_tensor_data(matrix, rows_indices, columns_indices, submatrix.data());

    EXPECT_LT((submatrix.array() - type(3.1416)).abs().maxCoeff(), type(1e-6));
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
    fill_tensor_data(matrix, rows, columns, submatrix.data());

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
    fill_tensor_data(matrix, rows, columns, submatrix.data());

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
    fill_tensor_data(matrix, rows, columns, submatrix.data());

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
    fill_tensor_data(matrix, rows, columns, memcpy_path.data(), 1);

    MatrixR gather_path(3, 2);
    fill_tensor_data(matrix, rows, columns, gather_path.data(), 0);

    EXPECT_LT((memcpy_path - gather_path).array().abs().maxCoeff(), type(1e-6));

    EXPECT_NEAR(memcpy_path(0, 0), type(1),  1e-6);
    EXPECT_NEAR(memcpy_path(2, 1), type(22), 1e-6);
}


TEST(Shape, DefaultIsEmpty)
{
    Shape shape;

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.rank, 0u);
    EXPECT_EQ(shape.size(), 0);
}


TEST(Shape, InitializerListConstructor)
{
    Shape shape{ 2, 3, 4 };

    EXPECT_FALSE(shape.empty());
    EXPECT_EQ(shape.rank, 3u);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.back(), 4);
    EXPECT_EQ(shape.size(), 24);
}


TEST(Shape, FillConstructor)
{
    Shape shape(size_t(3), Index(5));

    EXPECT_EQ(shape.rank, 3u);
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape[2], 5);
    EXPECT_EQ(shape.size(), 125);
}


TEST(Shape, Equality)
{
    EXPECT_EQ((Shape{ 2, 3 }), (Shape{ 2, 3 }));
    EXPECT_NE((Shape{ 2, 3 }), (Shape{ 2, 4 }));
    EXPECT_NE((Shape{ 2, 3 }), (Shape{ 2, 3, 1 }));
}


TEST(Shape, PushBackCapsAtMaxRank)
{
    Shape shape;

    shape.push_back(1);
    shape.push_back(2);

    EXPECT_EQ(shape.rank, 2u);
    EXPECT_EQ(shape.back(), 2);

    shape.push_back(3);
    shape.push_back(4);
    shape.push_back(5);

    EXPECT_EQ(shape.rank, Shape::MaxRank);
    EXPECT_EQ(shape[3], 4);
}


TEST(Shape, ClearResetsRank)
{
    Shape shape{ 2, 3, 4 };

    shape.clear();

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.size(), 0);
}


TEST(Shape, AppendCapsAtMaxRank)
{
    Shape shape{ 1, 2, 3 };

    shape.append(Shape{ 4, 5 });

    EXPECT_EQ(shape.rank, Shape::MaxRank);
    EXPECT_EQ(shape[3], 4);
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
    EXPECT_EQ(buffer.bytes, 0);
    EXPECT_EQ(buffer.size_in_floats(), 0);
    EXPECT_EQ(buffer.data, nullptr);
}


TEST(Buffer, ResizeBytesAllocatesAligned)
{
    Buffer buffer;

    buffer.resize_bytes(16, Device::CPU);

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.bytes, 16);
    EXPECT_EQ(buffer.size_in_floats(), 4);
    ASSERT_NE(buffer.data, nullptr);
    EXPECT_TRUE(is_aligned(buffer.data));
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
    EXPECT_GE(buffer.bytes, 4 * Index(sizeof(float)));

    data[0] = type(7);
    EXPECT_NEAR(buffer.as<float>()[0], type(7), 1e-6);
}


TEST(Buffer, GrowToOnlyGrows)
{
    Buffer buffer;
    buffer.resize_bytes(32, Device::CPU);
    EXPECT_EQ(buffer.bytes, 32);

    buffer.grow_to(16);
    EXPECT_EQ(buffer.bytes, 32);

    buffer.grow_to(64);
    EXPECT_EQ(buffer.bytes, 64);
}


TEST(Buffer, ResizeToZeroFrees)
{
    Buffer buffer;
    buffer.resize_bytes(16, Device::CPU);
    ASSERT_FALSE(buffer.empty());

    buffer.resize_bytes(0, Device::CPU);

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data, nullptr);
}


TEST(Buffer, MoveTransfersOwnership)
{
    Buffer source;
    source.resize_bytes(8 * Index(sizeof(float)), Device::CPU);
    source.as<float>()[0] = type(42);

    Buffer dest(move(source));

    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data, nullptr);

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


TEST(TensorView, ReshapePreservesDataPointer)
{
    Tensor2 storage(2, 3);
    storage.setZero();

    TensorView view(storage.data(), { 2, 3 });
    TensorView reshaped = view.reshape({ 3, 2 });

    EXPECT_EQ(reshaped.get_rank(), 2);
    EXPECT_EQ(reshaped.shape[0], 3);
    EXPECT_EQ(reshaped.shape[1], 2);
    EXPECT_EQ(reshaped.size(), 6);
    EXPECT_EQ(reshaped.as<type>(), view.as<type>());
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
