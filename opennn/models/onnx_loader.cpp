// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.
//
// YOLOv8s ONNX weight loader — no external dependencies.
// Reads an ONNX file exported by Ultralytics (yolo export format=onnx) and
// loads backbone + neck + box-branch weights into an OpenNN YOLOv8 network.
// Class-output layers are reinitialised to a 1 % prior for n_classes.

#include "opennn/models/models.h"
#include "opennn/core/log.h"
#include "opennn/network/layers/convolutional_layer.h"

#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace opennn
{

namespace
{

// ── Minimal protobuf wire decoder ─────────────────────────────────────────
//
// ONNX uses proto3.  We only need:
//   ModelProto  field 7  → GraphProto
//   GraphProto  field 5  → repeated TensorProto  (initialisers)
//   TensorProto field 1  → dims        (packed int64)
//               field 8  → name        (string)
//               field 9  → raw_data    (bytes → float32 LE)
//               field 4  → float_data  (packed float32)

static uint64_t read_varint(const uint8_t* buf, size_t& pos, size_t end)
{
    uint64_t result = 0;
    unsigned shift  = 0;
    while (pos < end)
    {
        const uint8_t b = buf[pos++];
        result |= uint64_t(b & 0x7f) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    return result;
}

static void skip_field(const uint8_t* buf, size_t& pos, size_t end, int wire_type)
{
    switch (wire_type)
    {
        case 0: read_varint(buf, pos, end);                                     break;
        case 1: pos += 8;                                                       break;
        case 2: { const auto n = read_varint(buf, pos, end); pos += size_t(n); break; }
        case 5: pos += 4;                                                       break;
        default: throw std::runtime_error("onnx_loader: unknown protobuf wire type "
                                          + std::to_string(wire_type));
    }
}

struct OnnxTensor
{
    string            name;
    vector<int64_t>   dims;
    vector<float>     data;  // always float32
};

static OnnxTensor parse_tensor_proto(const uint8_t* buf, size_t start, size_t end)
{
    OnnxTensor t;
    size_t pos = start;
    while (pos < end)
    {
        const uint64_t tag = read_varint(buf, pos, end);
        const int fn = int(tag >> 3);
        const int wt = int(tag & 7);

        if (fn == 1 && wt == 2)       // dims: packed int64
        {
            const auto len     = read_varint(buf, pos, end);
            const size_t sub_e = pos + size_t(len);
            while (pos < sub_e)
                t.dims.push_back(int64_t(read_varint(buf, pos, sub_e)));
        }
        else if (fn == 1 && wt == 0)  // dims: single int64
            t.dims.push_back(int64_t(read_varint(buf, pos, end)));
        else if (fn == 4 && wt == 2)  // float_data: packed float32
        {
            const auto len = read_varint(buf, pos, end);
            t.data.resize(size_t(len) / 4);
            memcpy(t.data.data(), buf + pos, size_t(len));
            pos += size_t(len);
        }
        else if (fn == 8 && wt == 2)  // name: string
        {
            const auto len = read_varint(buf, pos, end);
            t.name.assign(reinterpret_cast<const char*>(buf + pos), size_t(len));
            pos += size_t(len);
        }
        else if (fn == 9 && wt == 2)  // raw_data: bytes (LE float32)
        {
            const auto len = read_varint(buf, pos, end);
            t.data.resize(size_t(len) / 4);
            memcpy(t.data.data(), buf + pos, size_t(len));
            pos += size_t(len);
        }
        else
            skip_field(buf, pos, end, wt);
    }
    return t;
}

static map<string, OnnxTensor> parse_onnx_initializers(const uint8_t* buf, size_t total)
{
    map<string, OnnxTensor> result;
    size_t pos = 0;

    // ModelProto: scan top-level fields; enter field 7 (graph = GraphProto)
    while (pos < total)
    {
        const uint64_t tag = read_varint(buf, pos, total);
        const int fn = int(tag >> 3);
        const int wt = int(tag & 7);

        if (fn == 7 && wt == 2)  // graph
        {
            const auto glen = read_varint(buf, pos, total);
            const size_t g0 = pos;
            const size_t g1 = pos + size_t(glen);
            pos = g1;

            // GraphProto: collect field 6 (initializer = TensorProto)
            size_t gpos = g0;
            while (gpos < g1)
            {
                const uint64_t gtag = read_varint(buf, gpos, g1);
                const int gfn = int(gtag >> 3);
                const int gwt = int(gtag & 7);

                if (gfn == 5 && gwt == 2)
                {
                    const auto tlen = read_varint(buf, gpos, g1);
                    OnnxTensor t = parse_tensor_proto(buf, gpos, gpos + size_t(tlen));
                    gpos += size_t(tlen);
                    if (!t.name.empty())
                        result.emplace(t.name, std::move(t));
                }
                else
                    skip_field(buf, gpos, g1, gwt);
            }
        }
        else
            skip_field(buf, pos, total, wt);
    }
    return result;
}

// ── Architecture map ───────────────────────────────────────────────────────
//
// One entry per Convolutional layer in the YOLOv8s graph.
// Mirrors build_architecture() in yolov8s_to_nd.py exactly.

struct LayerEntry
{
    string label;     // OpenNN layer label
    string pt_key;    // Ultralytics state_dict key prefix
    int    cv1_half;  // 0 = full tensor, -1 = first half (cv1a), +1 = second half (cv1b)
    bool   has_bn;    // true = Conv+BN, false = Conv+bias (detection output)
    bool   skip;      // true = reinit (class output, n_classes differs)
};

static vector<LayerEntry> build_arch()
{
    // YOLOv8s depth repeats (D=0.33 rounded)
    const int d1=1, d2=2, d3=2, d4=1, nd=1;

    vector<LayerEntry> arch;

    auto cbn = [&](const string& label, const string& pt_key, int cv1_half = 0) {
        arch.push_back({label, pt_key, cv1_half, /*has_bn=*/true, /*skip=*/false});
    };
    auto cnb = [&](const string& label, const string& pt_key, bool skip = false) {
        arch.push_back({label, pt_key, 0, /*has_bn=*/false, skip});
    };

    // C2f block.  cv1a/cv1b share the same ONNX tensor, split along output channels.
    auto c2f = [&](const string& prefix, const string& pt, int n)
    {
        cbn(prefix + "_cv1a", pt + ".cv1", -1);
        cbn(prefix + "_cv1b", pt + ".cv1", +1);
        for (int j = 0; j < n; ++j)
        {
            const string bp = prefix + "_b" + std::to_string(j + 1);
            const string pp = pt + ".m." + std::to_string(j);
            cbn(bp + "_cv1", pp + ".cv1");
            cbn(bp + "_cv2", pp + ".cv2");
        }
        cbn(prefix + "_cv2", pt + ".cv2");
    };

    // Backbone
    cbn("c8_stem",    "model.0");
    cbn("c8_s1_down", "model.1");
    c2f("c8_s1", "model.2", d1);
    cbn("c8_s2_down", "model.3");
    c2f("c8_s2", "model.4", d2);
    cbn("c8_s3_down", "model.5");
    c2f("c8_s3", "model.6", d3);
    cbn("c8_s4_down", "model.7");
    c2f("c8_s4", "model.8", d4);

    // SPPF
    cbn("c8_sppf_in",  "model.9.cv1");
    cbn("c8_sppf_out", "model.9.cv2");

    // FPN / PANet neck
    c2f("c8_n12", "model.12", nd);
    c2f("c8_n15", "model.15", nd);
    cbn("c8_pan_n4_down", "model.16");
    c2f("c8_n18", "model.18", nd);
    cbn("c8_pan_n5_down", "model.19");
    c2f("c8_n21", "model.21", nd);

    // Detection heads (small=stride 8 / medium=stride 16 / large=stride 32)
    const char* const heads[] = {"small", "medium", "large"};
    for (int i = 0; i < 3; ++i)
    {
        const string p = "c8_" + string(heads[i]);
        const string d = "model.22";
        cbn(p + "_box_c1", d + ".cv2." + std::to_string(i) + ".0");
        cbn(p + "_box_c2", d + ".cv2." + std::to_string(i) + ".1");
        cnb(p + "_box_out", d + ".cv2." + std::to_string(i) + ".2");
        cbn(p + "_cls_c1", d + ".cv3." + std::to_string(i) + ".0");
        cbn(p + "_cls_c2", d + ".cv3." + std::to_string(i) + ".1");
        cnb(p + "_cls_out", d + ".cv3." + std::to_string(i) + ".2", /*skip=*/true);
    }

    return arch;
}

} // anonymous namespace

// ── Public API ────────────────────────────────────────────────────────────

Index load_yolov8s_onnx(Network& network,
                        const filesystem::path& onnx_path,
                        Index n_classes)
{
    // 1. Read file
    ifstream f(onnx_path, ios::binary | ios::ate);
    if (!f)
        throw runtime_error("load_yolov8s_onnx: cannot open " + onnx_path.string());

    const auto bytes = static_cast<size_t>(f.tellg());
    vector<uint8_t> raw;
    raw.resize(bytes);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(bytes));

    // 2. Parse ONNX initialisers
    const map<string, OnnxTensor> tensors = parse_onnx_initializers(raw.data(), bytes);
    logging::info() << "load_yolov8s_onnx: parsed " << tensors.size()
                    << " initialisers from " << onnx_path << "\n";

    // 3. Walk architecture map and write each layer
    const vector<LayerEntry> arch = build_arch();
    static constexpr float PRIOR_BIAS = -4.5951f;

    Index loaded = 0;
    for (const LayerEntry& e : arch)
    {
        Layer* base_layer = nullptr;
        try { base_layer = network.get_layer(e.label).get(); }
        catch (...)
        {
            logging::warning() << "load_yolov8s_onnx: layer \"" << e.label
                               << "\" not in network — skipping\n";
            continue;
        }

        auto* conv = dynamic_cast<Convolutional*>(base_layer);
        if (!conv)
        {
            logging::warning() << "load_yolov8s_onnx: \"" << e.label
                               << "\" is not Convolutional — skipping\n";
            continue;
        }

        // ── Class-output: reinit ───────────────────────────────────────────
        if (e.skip)
        {
            conv->reinit_onnx_cls_out(PRIOR_BIAS);
            logging::info() << "  REINIT  " << e.label << "\n";
            ++loaded;
            continue;
        }

        // ── Conv + BN (BN folded into conv.bias in ONNX export) ───────────
        if (e.has_bn)
        {
            const string wk = e.pt_key + ".conv.weight";
            const string bk = e.pt_key + ".conv.bias";
            const auto it_w = tensors.find(wk);
            const auto it_b = tensors.find(bk);
            if (it_w == tensors.end() || it_b == tensors.end())
            {
                logging::warning() << "load_yolov8s_onnx: \"" << wk
                                   << "\" not found — skipping " << e.label << "\n";
                continue;
            }

            // cv1 split: ONNX tensor has 2*oc output channels; take first or second half.
            // Element offset in the NCHW flattened buffer is (oc * ic * kH * kW).
            const Index oc = conv->get_kernels_number();
            const Index ic = conv->get_kernel_channels();
            const Index kH = conv->get_kernel_height();
            const Index kW = conv->get_kernel_width();
            const Index k_offset = (e.cv1_half == +1) ? oc * ic * kH * kW : 0;
            const Index b_offset = (e.cv1_half == +1) ? oc : 0;

            conv->load_onnx_folded_conv_bn(
                it_w->second.data.data() + k_offset,
                it_b->second.data.data() + b_offset);

            logging::info() << "  loaded  " << e.label << "  ← " << wk << "\n";
        }
        // ── Conv + bias (no BN) ────────────────────────────────────────────
        else
        {
            const string wk = e.pt_key + ".weight";
            const auto it_w = tensors.find(wk);
            if (it_w == tensors.end())
            {
                logging::warning() << "load_yolov8s_onnx: \"" << wk
                                   << "\" not found — skipping " << e.label << "\n";
                continue;
            }

            conv->load_onnx_conv_bias(
                tensors.at(e.pt_key + ".bias").data.data(),
                it_w->second.data.data());

            logging::info() << "  loaded  " << e.label << "  ← " << wk << "\n";
        }

        ++loaded;
    }

    logging::info() << "load_yolov8s_onnx: loaded "
                    << loaded << "/" << arch.size() << " layers\n";
    return loaded;
}

} // namespace opennn
