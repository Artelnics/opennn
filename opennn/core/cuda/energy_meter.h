// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

namespace opennn::device
{

// Board power over a window, read through NVML, for the autotuners that
// choose between kernels of comparable speed by what they cost to run.
//
// NVML is loaded at run time (libnvidia-ml.so.1 ships with the driver, not
// the toolkit), so the library links nothing extra and a machine without a
// working NVML simply reports the meter unavailable, which the callers treat
// as "choose by time alone". The device is matched to the current CUDA
// device by PCI bus id, not by index: NVML and CUDA enumerate GPUs in
// different orders.
//
// The reading is the driver's own power sample ring (nvmlDeviceGetSamples,
// one sample per ~20 ms), averaged over the samples that fall inside the
// window, rather than the cumulative energy counter, which this GeForce
// reports as zero, or nvmlDeviceGetPowerUsage, which is a one-second
// average and cannot tell two kernels apart. A window needs to hold several
// samples to mean anything; energy_meter_window_watts() reports how many it
// held so the caller can refuse a reading that is too thin.
bool energy_meter_available() noexcept;

// A timestamp marking the start of a window, in NVML's clock (microseconds).
unsigned long long energy_meter_begin() noexcept;

// Mean board power in watts over the samples taken since `begin`, and the
// number of samples that mean rests on. Returns 0 W with 0 samples when the
// meter is unavailable or the ring held nothing newer than `begin`.
double energy_meter_window_watts(unsigned long long begin, int& samples) noexcept;

}
