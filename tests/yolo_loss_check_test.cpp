#include "pch.h"

#ifndef OPENNN_NO_VISION

#include "opennn/training_strategy/loss.h"

#include <cmath>
#include <vector>

using namespace opennn;

TEST(YoloLossCheck, GradientMatchesFiniteDifferences)
{
    constexpr int gc_grid = 2;
    constexpr int gc_bpc  = 2;
    constexpr int gc_ncls = 3;
    constexpr int gc_vpb  = 5 + gc_ncls;
    constexpr int gc_N    = gc_grid * gc_grid * gc_bpc * gc_vpb;

    vector<float> gc_out(gc_N, 0.0f);
    vector<float> gc_tgt(gc_N, 0.0f);
    vector<float> gc_grad(gc_N, 0.0f);

    for (int i = 0; i < gc_N; ++i)
        gc_out[i] = 0.5f;

    gc_out[0] = 0.52f;
    gc_out[1] = 0.53f;
    gc_out[2] = 0.22f;
    gc_out[3] = 0.17f;
    gc_out[4] = 0.70f;
    gc_out[5] = 0.80f;
    gc_out[6] = 0.15f;
    gc_out[7] = 0.05f;

    gc_tgt[0] = 0.5f;
    gc_tgt[1] = 0.5f;
    gc_tgt[2] = 0.2f;
    gc_tgt[3] = 0.15f;
    gc_tgt[4] = 1.0f;
    gc_tgt[5] = 1.0f;

    const Shape gc_shape({1, Index(gc_grid), Index(gc_grid), Index(gc_bpc * gc_vpb)});
    TensorView gc_out_tv (gc_out.data(),  gc_shape, Type::FP32);
    TensorView gc_tgt_tv (gc_tgt.data(),  gc_shape, Type::FP32);
    TensorView gc_grad_tv(gc_grad.data(), gc_shape, Type::FP32);

    const YoloLambdas gc_lam{5.0f, 0.5f, 2.0f, 0.0f};
    const float gc_inv_batch = 1.0f;

    yolo_gradient_kernel(gc_out_tv, gc_tgt_tv, gc_grad_tv,
                         Index(gc_bpc), Index(gc_ncls),
true, gc_inv_batch, gc_lam);

    const float gc_eps = 1e-4f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < gc_N; ++i)
    {
        const float orig = gc_out[i];
        gc_out[i] = orig + gc_eps;
        const float lp = yolo_error_kernel(gc_out_tv, gc_tgt_tv,
                                           Index(gc_bpc), Index(gc_ncls),
true, gc_lam);
        gc_out[i] = orig - gc_eps;
        const float lm = yolo_error_kernel(gc_out_tv, gc_tgt_tv,
                                           Index(gc_bpc), Index(gc_ncls),
true, gc_lam);
        gc_out[i] = orig;

        const float num_grad = (lp - lm) / (2.0f * gc_eps);
        const float ana_grad = gc_grad[i];
        const float denom    = max(abs(num_grad), abs(ana_grad)) + 1e-8f;
        const float rel      = abs(num_grad - ana_grad) / denom;
        if (rel > max_rel_err) max_rel_err = rel;
    }

    EXPECT_LT(max_rel_err, 1e-2f);
}

TEST(YoloLossCheck, ForwardMatchesExpectedValues)
{

    constexpr int ev_grid = 1;
    constexpr int ev_bpc  = 1;
    constexpr int ev_ncls = 2;
    constexpr int ev_vpb  = 5 + ev_ncls;

    const float ev_cx=0.5f, ev_cy=0.5f, ev_w=0.4f, ev_h=0.3f;
    const float ev_conf=0.7f, ev_p0=0.8f, ev_p1=0.2f;
    const YoloLambdas ev_lam{5.0f, 0.5f, 2.0f, 0.0f};

    vector<float> ev_out_A(ev_vpb, 0.0f);
    vector<float> ev_tgt_A(ev_vpb, 0.0f);
    ev_out_A[0]=ev_cx; ev_out_A[1]=ev_cy; ev_out_A[2]=ev_w; ev_out_A[3]=ev_h;
    ev_out_A[4]=ev_conf; ev_out_A[5]=ev_p0; ev_out_A[6]=ev_p1;
    ev_tgt_A[0]=ev_cx; ev_tgt_A[1]=ev_cy; ev_tgt_A[2]=ev_w; ev_tgt_A[3]=ev_h;
    ev_tgt_A[4]=1.0f;
    ev_tgt_A[5]=1.0f; ev_tgt_A[6]=0.0f;

    const float EPSILON_LOCAL = 1e-7f;

    const float expA = 0.0f
        + (-std::log(ev_conf + EPSILON_LOCAL))
        + ev_lam.cls * (-std::log(ev_p0 + EPSILON_LOCAL) - std::log(1.0f - ev_p1 + EPSILON_LOCAL));

    const Shape ev_shape_A({1, Index(ev_grid), Index(ev_grid), Index(ev_bpc * ev_vpb)});
    TensorView ev_out_tv_A(ev_out_A.data(), ev_shape_A, Type::FP32);
    TensorView ev_tgt_tv_A(ev_tgt_A.data(), ev_shape_A, Type::FP32);
    const float gotA = yolo_error_kernel(ev_out_tv_A, ev_tgt_tv_A,
                                         Index(ev_bpc), Index(ev_ncls),
true, ev_lam);
    const float errA = abs(gotA - expA);

    const float ev_px=0.1f, ev_gx=0.9f, ev_py=0.5f, ev_gy=0.5f;
    const float ev_pw=0.1f, ev_gw=0.1f, ev_ph=0.1f, ev_gh=0.1f;

    vector<float> ev_out_B(ev_vpb, 0.0f);
    vector<float> ev_tgt_B(ev_vpb, 0.0f);
    ev_out_B[0]=ev_px; ev_out_B[1]=ev_py; ev_out_B[2]=ev_pw; ev_out_B[3]=ev_ph;
    ev_out_B[4]=0.5f; ev_out_B[5]=0.6f; ev_out_B[6]=0.4f;
    ev_tgt_B[0]=ev_gx; ev_tgt_B[1]=ev_gy; ev_tgt_B[2]=ev_gw; ev_tgt_B[3]=ev_gh;
    ev_tgt_B[4]=1.0f; ev_tgt_B[5]=1.0f; ev_tgt_B[6]=0.0f;

    const float ev_enc=0.09f, ev_uni=0.02f;
    const float ev_giou = 0.0f - (ev_enc - ev_uni) / ev_enc;
    const float ev_dx = ev_px - ev_gx, ev_dy = ev_py - ev_gy;
    const float ev_ew = 0.90f, ev_eh = 0.10f;
    const float ev_c2 = ev_ew*ev_ew + ev_eh*ev_eh + EPSILON_LOCAL;
    const float ev_ciou = ev_giou - (ev_dx*ev_dx + ev_dy*ev_dy) / ev_c2;
    const float expB_coord = ev_lam.giou * (1.0f - ev_ciou);

    const float expB_obj   = -std::log(ev_out_B[4] + EPSILON_LOCAL);
    const float expB_cls   = ev_lam.cls * (-std::log(ev_out_B[5] + EPSILON_LOCAL)
                                           - std::log(1.0f - ev_out_B[6] + EPSILON_LOCAL));
    const float expB       = expB_coord + expB_obj + expB_cls;

    TensorView ev_out_tv_B(ev_out_B.data(), ev_shape_A, Type::FP32);
    TensorView ev_tgt_tv_B(ev_tgt_B.data(), ev_shape_A, Type::FP32);
    const float gotB = yolo_error_kernel(ev_out_tv_B, ev_tgt_tv_B,
                                         Index(ev_bpc), Index(ev_ncls),
true, ev_lam);
    const float errB = abs(gotB - expB);

    vector<float> ev_out_C = ev_out_A;
    ev_out_C[4] = 0.1f;
    vector<float> ev_grad_C(ev_vpb, 0.0f);
    TensorView ev_out_tv_C(ev_out_C.data(), ev_shape_A, Type::FP32);
    TensorView ev_grad_tv_C(ev_grad_C.data(), ev_shape_A, Type::FP32);
    yolo_gradient_kernel(ev_out_tv_C, ev_tgt_tv_A, ev_grad_tv_C,
                         Index(ev_bpc), Index(ev_ncls), true, 1.0f, ev_lam);

    const float ev_raw_logit_grad_obj = ev_grad_C[4] * ev_out_C[4] * (1.0f - ev_out_C[4]);

    const float errC = (ev_raw_logit_grad_obj < 0.0f) ? 0.0f : 1.0f;

    vector<float> ev_out_D(ev_vpb, 0.0f);
    vector<float> ev_tgt_D(ev_vpb, 0.0f);
    ev_out_D[4] = 0.9f;
    vector<float> ev_grad_D(ev_vpb, 0.0f);
    TensorView ev_out_tv_D(ev_out_D.data(), ev_shape_A, Type::FP32);
    TensorView ev_tgt_tv_D(ev_tgt_D.data(), ev_shape_A, Type::FP32);
    TensorView ev_grad_tv_D(ev_grad_D.data(), ev_shape_A, Type::FP32);
    yolo_gradient_kernel(ev_out_tv_D, ev_tgt_tv_D, ev_grad_tv_D,
                         Index(ev_bpc), Index(ev_ncls), true, 1.0f, ev_lam);
    const float ev_raw_logit_grad_bg = ev_grad_D[4] * ev_out_D[4] * (1.0f - ev_out_D[4]);
    const float errD = (ev_raw_logit_grad_bg > 0.0f) ? 0.0f : 1.0f;

    EXPECT_NEAR(gotA, expA, 1e-4f) << "perfect overlap (coord_loss=0)";
    EXPECT_NEAR(gotB, expB, 1e-4f) << "non-overlap CIoU";
    EXPECT_LT(ev_raw_logit_grad_obj, 0.0f) << "foreground obj gradient direction";
    EXPECT_GT(ev_raw_logit_grad_bg, 0.0f) << "background obj gradient direction";

    EXPECT_LT(max({errA, errB, errC, errD}), 1e-4f);
}

#endif
