#!/usr/bin/env python3
"""
Convert official YOLOv8s backbone + neck weights into the opennn binary format.

What transfers:
  - Backbone (model.0–model.9 incl. SPPF): all weights
  - Neck C2f + PAN convs (model.12, 15, 16, 18, 19, 21): all weights
  - Detection box branch (model.22.cv2): box_c1, box_c2 (kernel + BN), box_out kernel
  - Detection cls branch (model.22.cv3): cls_c1, cls_c2 (kernel + BN) — 128ch matches official
  - DFL conv (model.22.dfl): NOT loaded (embedded in DetectionV8 layer, no explicit param)

What is zeroed / prior-init (architecture differences vs official):
  - cls_out: n_classes != 80 — bias = -4.5951 (1% prior), kernel = 0
  - box_out bias: official has bias, opennn does not (add_conv with use_bias=false)

Kernel layout conversion: Ultralytics [N,C,H,W] → opennn [N,H,W,C] (permute 0,2,3,1).
C2f cv1 split: official cv1 [2*half, in, 1,1] → first half → cv1a, second half → cv1b.

Output: versioned OPENNNP binary with correct header (so layout fingerprint is checked).
        Written as LEGACY raw FP32 (no header) so it loads via the size-match path.

Usage:
    python3 yolov8s_to_nd.py [--pt yolov8s.pt] [--out yolo_data_v8-pretrained] [--classes N]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


ALIGN_ELEMENTS = 16  # EIGEN_MAX_ALIGN_BYTES / sizeof(float) = 64/4 on this machine


def align_up(n, a=ALIGN_ELEMENTS):
    return (n + a - 1) & ~(a - 1)


def padded(arr: np.ndarray) -> np.ndarray:
    """Pad arr to next ALIGN_ELEMENTS boundary with zeros."""
    n = len(arr)
    p = align_up(n)
    if p == n:
        return arr
    return np.concatenate([arr, np.zeros(p - n, dtype=np.float32)])


def to_nhwc(w: np.ndarray) -> np.ndarray:
    """[N,C,H,W] → [N,H,W,C]"""
    return w.transpose(0, 2, 3, 1) if w.ndim == 4 else w


def load_conv_bn(sd, pt_key, expected_oc, expected_ic,
                 cv1_slice=None, silent_mismatch=False):
    """
    Load kernel + BN (gamma, beta, running_mean, running_var) from state_dict.

    expected_oc / expected_ic: what opennn expects (used to detect shape mismatches).
    cv1_slice: "a" or "b" → take first/second half of the cv1 output channels.
    silent_mismatch: if True and shapes differ, return zero arrays instead of raising.

    Returns (kernel, gamma, beta, rmean, rvar) all np.float32, or None if mismatch.
    """
    w_key  = pt_key + ".conv.weight"
    g_key  = pt_key + ".bn.weight"
    b_key  = pt_key + ".bn.bias"
    rm_key = pt_key + ".bn.running_mean"
    rv_key = pt_key + ".bn.running_var"

    if w_key not in sd:
        raise KeyError(f"Missing {w_key}")

    k = sd[w_key].float().numpy()   # [out, in, H, W]

    if cv1_slice is not None:
        half = k.shape[0] // 2
        k = k[:half] if cv1_slice == "a" else k[half:]

    # Shape validation
    actual_oc = k.shape[0]
    actual_ic = k.shape[1]
    if actual_oc != expected_oc or actual_ic != expected_ic:
        if silent_mismatch:
            return None
        raise ValueError(
            f"{pt_key}: expected ({expected_oc}, {expected_ic}, *, *), "
            f"got ({actual_oc}, {actual_ic}, *, *)")

    k = to_nhwc(k)

    g  = sd[g_key].float().numpy()
    b  = sd[b_key].float().numpy()
    rm = sd[rm_key].float().numpy()
    rv = sd[rv_key].float().numpy()

    if cv1_slice is not None:
        half = len(g) // 2
        sl   = slice(None, half) if cv1_slice == "a" else slice(half, None)
        g, b, rm, rv = g[sl], b[sl], rm[sl], rv[sl]

    return k.astype(np.float32), g.astype(np.float32), \
           b.astype(np.float32), rm.astype(np.float32), rv.astype(np.float32)


def load_conv_bias(sd, pt_key, expected_oc, expected_ic, silent_mismatch=False):
    """
    Load bias + kernel for a conv-without-BN layer (opennn sets use_bias=True when no BN).
    The official Ultralytics detect head convs have both .weight and .bias tensors.
    Returns (bias, kernel) or None if shape mismatch.
    """
    w_key = pt_key + ".weight"
    b_key = pt_key + ".bias"
    if w_key not in sd:
        raise KeyError(f"Missing {w_key}")
    k = sd[w_key].float().numpy()
    if k.shape[0] != expected_oc or k.shape[1] != expected_ic:
        if silent_mismatch:
            return None
        raise ValueError(f"{pt_key}: shape mismatch")
    k = to_nhwc(k).astype(np.float32)
    b = sd[b_key].float().numpy().astype(np.float32) if b_key in sd else np.zeros(expected_oc, np.float32)
    return b, k


def build_architecture(n_classes, reg_max=16):
    """
    Return a flat list of layer descriptors for the opennn CSPDarknet53v11 + FPNv8
    network at YOLOv8s scale (width=0.5, depth=0.33).

    Each dict has:
      type     : "conv_bn" | "conv_nobias" | "no_param"
      label    : opennn layer label
      oc, ic   : output / input channels
      kh, kw   : kernel height / width
      pt_key   : Ultralytics state_dict prefix (conv_bn / conv_nobias types only)
      cv1_slice: "a" | "b" | None
      skip     : True → zero-initialise instead of loading (architecture mismatch)
    """

    def r8(x):
        return max(8, int(round(x / 8)) * 8)

    W, D = 0.5, 0.33
    c1 = r8(64   * W)   # 32
    c2 = r8(128  * W)   # 64
    c3 = r8(256  * W)   # 128
    c4 = r8(512  * W)   # 256
    c5 = r8(1024 * W)   # 512
    d1 = max(1, round(3 * D))   # 1
    d2 = max(1, round(6 * D))   # 2
    d3 = max(1, round(6 * D))   # 2
    d4 = max(1, round(3 * D))   # 1
    n12 = r8(512  * W)  # 256
    n15 = r8(256  * W)  # 128
    n18 = r8(512  * W)  # 256
    n21 = r8(1024 * W)  # 512
    nd  = max(1, round(3 * D))  # 1

    box_head_ch = 64
    cls_head_ch = 128  # matches official YOLOv8s cv3 branch width
    box_ch  = 4 * max(reg_max, 1)

    layers = []

    def cbn(label, oc, kh, kw, ic, pt_key, cv1_slice=None, skip=False):
        layers.append(dict(type="conv_bn", label=label,
                           oc=oc, ic=ic, kh=kh, kw=kw,
                           pt_key=pt_key, cv1_slice=cv1_slice, skip=skip))
        layers.append(dict(type="no_param", label=label + "_act"))

    def cnb(label, oc, kh, kw, ic, pt_key, skip=False):
        # No BN → opennn sets use_bias=True; params are [bias{oc}, kernel{oc,kh,kw,ic}]
        layers.append(dict(type="conv_bias", label=label,
                           oc=oc, ic=ic, kh=kh, kw=kw,
                           pt_key=pt_key, skip=skip))

    def nop(label):
        layers.append(dict(type="no_param", label=label))

    def c2f(prefix, pt, in_ch, out_ch, n, shortcut):
        half = out_ch // 2
        cbn(prefix+"_cv1a", half, 1, 1, in_ch, pt+".cv1", cv1_slice="a")
        cbn(prefix+"_cv1b", half, 1, 1, in_ch, pt+".cv1", cv1_slice="b")
        for j in range(n):
            bp = prefix + f"_b{j+1}"
            pp = pt + f".m.{j}"
            cbn(bp+"_cv1", half, 3, 3, half, pp+".cv1")
            cbn(bp+"_cv2", half, 3, 3, half, pp+".cv2")
            if shortcut:
                nop(bp + "_add")
        nop(prefix + "_cat")
        cbn(prefix+"_cv2", out_ch, 1, 1, (2+n)*half, pt+".cv2")

    # Backbone
    cbn("c8_stem",    c1, 3, 3, 3,   "model.0")
    cbn("c8_s1_down", c2, 3, 3, c1,  "model.1")
    c2f("c8_s1", "model.2", c2, c2, d1, shortcut=True)
    cbn("c8_s2_down", c3, 3, 3, c2,  "model.3")
    c2f("c8_s2", "model.4", c3, c3, d2, shortcut=True)
    cbn("c8_s3_down", c4, 3, 3, c3,  "model.5")
    c2f("c8_s3", "model.6", c4, c4, d3, shortcut=True)
    cbn("c8_s4_down", c5, 3, 3, c4,  "model.7")
    c2f("c8_s4", "model.8", c5, c5, d4, shortcut=True)

    # SPPF
    hs = c5 // 2
    cbn("c8_sppf_in",  hs, 1, 1, c5,   "model.9.cv1")
    nop("c8_sppf_p1"); nop("c8_sppf_p2"); nop("c8_sppf_p3"); nop("c8_sppf_cat")
    cbn("c8_sppf_out", c5, 1, 1, 4*hs, "model.9.cv2")

    # FPN / PANet neck
    nop("c8_fpn_p5_up"); nop("c8_fpn_p4_cat")
    c2f("c8_n12", "model.12", c5+c4,     n12, nd, shortcut=False)
    nop("c8_fpn_p4_up"); nop("c8_fpn_p3_cat")
    c2f("c8_n15", "model.15", n12+c3,    n15, nd, shortcut=False)
    cbn("c8_pan_n4_down", n15, 3, 3, n15, "model.16")
    nop("c8_pan_n4_cat")
    c2f("c8_n18", "model.18", n15+n12,   n18, nd, shortcut=False)
    cbn("c8_pan_n5_down", n18, 3, 3, n18, "model.19")
    nop("c8_pan_n5_cat")
    c2f("c8_n21", "model.21", n18+c5,    n21, nd, shortcut=False)

    # Detection heads
    # cls branch now uses cls_head_ch=128 matching official YOLOv8s cv3 → can load
    heads_in = [n15, n18, n21]
    heads_name = ["small", "medium", "large"]
    for i, (hname, inch) in enumerate(zip(heads_name, heads_in)):
        p = f"c8_{hname}"
        d = f"model.22"
        # Box branch: 64ch → load from cv2
        cbn (p+"_box_c1",  box_head_ch, 3, 3, inch,        f"{d}.cv2.{i}.0")
        cbn (p+"_box_c2",  box_head_ch, 3, 3, box_head_ch, f"{d}.cv2.{i}.1")
        cnb (p+"_box_out", box_ch,      1, 1, box_head_ch, f"{d}.cv2.{i}.2")
        # Cls branch: 128ch matches official cv3 → load c1/c2; skip cls_out (n_classes differs)
        cbn (p+"_cls_c1",  cls_head_ch, 3, 3, inch,        f"{d}.cv3.{i}.0")
        cbn (p+"_cls_c2",  cls_head_ch, 3, 3, cls_head_ch, f"{d}.cv3.{i}.1")
        cnb (p+"_cls_out", n_classes,   1, 1, cls_head_ch, f"{d}.cv3.{i}.2", skip=True)
        nop(p+"_cat"); nop(p+"_det")

    return layers


def convert(args):
    # ── Load official model ────────────────────────────────────────────────
    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"{pt_path} not found — downloading yolov8s.pt …")
        try:
            from ultralytics import YOLO
            YOLO("yolov8s.pt")
            pt_path = Path("yolov8s.pt")
        except Exception as e:
            sys.exit(f"Download failed: {e}\n"
                     "Place yolov8s.pt next to this script and retry.")

    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    print(f"Loaded {len(sd)} tensors from {pt_path}")

    # ── Build architecture map ─────────────────────────────────────────────
    arch   = build_architecture(n_classes=args.classes, reg_max=args.reg_max)
    params = []
    states = []
    log    = []

    PRIOR_BIAS = -4.5951  # class output prior (as in C++ set_parameters_random)

    def emit(arr):
        """Append arr padded to ALIGN_ELEMENTS boundary."""
        params.append(padded(arr.flatten().astype(np.float32)))

    def emit_state(arr):
        states.append(padded(arr.flatten().astype(np.float32)))

    for L in arch:
        t = L["type"]
        if t == "no_param":
            continue

        label = L["label"]
        oc, ic, kh, kw = L["oc"], L["ic"], L["kh"], L["kw"]

        if L.get("skip"):
            if t == "conv_bias":
                # cls_out: prior bias so initial confidence = 1%; kernel = 0 (not PRIOR_BIAS,
                # which would push all logits to large negative and block learning).
                bias_fill = PRIOR_BIAS if label.endswith("_cls_out") else 0.0
                emit(np.full(oc, bias_fill, np.float32))
                emit(np.zeros(oc * kh * kw * ic, np.float32))
            elif t == "conv_bn":
                # cls_c1 / cls_c2: Kaiming uniform so pretrained backbone features immediately
                # flow into the class branch rather than being blocked by zero weights.
                fan_in = ic * kh * kw
                bound = np.sqrt(1.0 / fan_in)
                emit(np.random.uniform(-bound, bound, oc * kh * kw * ic).astype(np.float32))
                emit(np.ones(oc, np.float32))                 # gamma
                emit(np.zeros(oc, np.float32))                # beta
                emit_state(np.zeros(oc, np.float32))          # running_mean
                emit_state(np.ones(oc, np.float32))           # running_var
            log.append(f"  INIT    {label}  (architecture mismatch — random init)")
            continue

        pt_key    = L["pt_key"]
        cv1_slice = L.get("cv1_slice")

        if t == "conv_bn":
            result = load_conv_bn(sd, pt_key, oc, ic,
                                  cv1_slice=cv1_slice, silent_mismatch=True)
            if result is None:
                emit(np.zeros(oc * kh * kw * ic, np.float32))
                emit(np.ones(oc, np.float32))
                emit(np.zeros(oc, np.float32))
                emit_state(np.zeros(oc, np.float32))
                emit_state(np.ones(oc, np.float32))
                log.append(f"  ZEROED  {label}  (shape mismatch for {pt_key})")
            else:
                k, g, b, rm, rv = result
                emit(k); emit(g); emit(b)
                emit_state(rm); emit_state(rv)
                log.append(f"  loaded  {label}  ← {pt_key}")

        elif t == "conv_bias":
            # opennn: no BN → use_bias=True → params = [bias{oc}, kernel{oc,kh,kw,ic}]
            result = load_conv_bias(sd, pt_key, oc, ic, silent_mismatch=True)
            if result is None:
                emit(np.zeros(oc, np.float32))                # bias
                emit(np.zeros(oc * kh * kw * ic, np.float32))  # kernel
                log.append(f"  ZEROED  {label}  (shape mismatch for {pt_key})")
            else:
                bias, kernel = result
                emit(bias); emit(kernel)
                log.append(f"  loaded  {label}  ← {pt_key}")

    print("\n".join(log))

    all_params = np.concatenate(params).astype(np.float32)
    all_states = np.concatenate(states).astype(np.float32)

    print(f"\nGenerated {len(all_params):,} param floats ({len(all_params)*4/1024/1024:.2f} MB)")
    print(f"Generated {len(all_states):,} state floats ({len(all_states)*4/1024:.1f} KB)")

    # ── Write output ───────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem   = f"yolo_weights_{args.tag}_c11_fpnv8_sppf_sigmoid_dfl_s_bce_ig_bgfocal"
    w_path = out_dir / (stem + ".bin")
    s_path = out_dir / (stem + ".states.bin")
    e_path = out_dir / (stem + "_epochs.txt")

    w_path.write_bytes(all_params.tobytes())
    s_path.write_bytes(all_states.tobytes())
    e_path.write_text("0\n")

    print(f"\nWrote params  → {w_path}  ({w_path.stat().st_size:,} bytes)")
    print(f"Wrote states  → {s_path}  ({s_path.stat().st_size:,} bytes)")
    print(f"Reset epochs  → {e_path}")

    # ── Sanity check: compare against a same-class reference if available ──
    ref = out_dir.parent / "yolo_data_v8-scratch" / (stem + ".bin")
    our_bytes = w_path.stat().st_size
    if ref.exists():
        ref_bytes = ref.stat().st_size
        # Reference is always 3-class; only compare directly when classes == 3
        if args.classes == 3:
            if ref_bytes == our_bytes:
                print(f"\n✓ Size matches 3-class reference ({ref_bytes:,} bytes)")
            else:
                print(f"\n✗ Size mismatch! Expected {ref_bytes:,}, got {our_bytes:,}")
                print("  The opennn binary will reject this file (size-based fingerprint).")
                print("  Check for architecture description errors in build_architecture().")
        else:
            # Compute expected extra bytes: 3 cls_out layers each gain
            # (padded(n_classes) - padded(3)) bias floats + (n_classes*128 - 3*128) kernel floats
            def pad16(n): return (n + 15) & ~15
            extra = 3 * ((pad16(args.classes) + args.classes * 128)
                         - (pad16(3) + 3 * 128)) * 4
            expected = ref_bytes - 56 + extra  # strip header from ref (versioned), add class delta
            if our_bytes == expected:
                print(f"\n✓ Size correct for {args.classes} classes ({our_bytes:,} bytes)")
            else:
                print(f"\n✗ Size mismatch! Expected {expected:,}, got {our_bytes:,}")
                print("  Check build_architecture() for n_classes handling.")
    else:
        print(f"\n  Output: {our_bytes:,} bytes (no reference available for size check)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt",      default="yolov8s.pt")
    ap.add_argument("--out",     default="yolo_data_v8-pretrained")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--tag",     default="synth", help="dataset tag used in weight filename (e.g. synth, voc, coco)")
    ap.add_argument("--reg-max", type=int, default=16)
    args = ap.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
