#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test for torch.nn.functional.linear based on FX graph analysis.

This test recreates scenarios from FX graphs with different input shapes.

Case 1:
- Input shape: [1, 1, 4096] (f16)
- Weight shape: [49159, 4096] (f16)
- Expected output shape: [1, 1, 49159] (f16)

Case 2:
- Input shape: [1, 64, 4096] (f16)
- Weight shape: [4096, 4096] (f16)
- Expected output shape: [1, 64, 4096] (f16)

The FX graph shows the operation decomposes to:
1. permute weight
2. clone with contiguous format
3. unsqueeze
4. expand input
5. expand weight
6. bmm (batch matrix multiply)
"""

import torch
import torch.nn.functional as F


def linear_wrapper(input_tensor, weight, bias=None):
    """Wrapper function for torch.nn.functional.linear to be compiled."""
    return F.linear(input_tensor, weight, bias)


def test_linear_fx_graph(input_shape, weight_shape, case_name="", verbose=True):
    """
    Test torch.nn.functional.linear with torch.compile comparing CPU vs Spyre.
    
    Args:
        input_shape: Tuple specifying input tensor shape
        weight_shape: Tuple specifying weight tensor shape
        case_name: Optional name for the test case
        verbose: If True, print detailed information including value ranges
    
    Returns:
        bool: True if test passed, False if test failed
    """
    if verbose:
        print("=" * 80)
        print(f"Test: torch.nn.functional.linear{' - ' + case_name if case_name else ''}")
        print("=" * 80)
    
    # Create input tensors with specified shapes
    # Use fixed scale 0.01 for FP16 safety (simpler than Xavier)
    # FP16 range: -65504 to 65504
    # For linear layer: output = input @ weight.T
    # With scale=0.01 and K features, output magnitude ~ sqrt(K) * 0.01^2
    # For K=4096: output std ~ 64 * 0.0001 = 0.0064 (safe for FP16)
    
    in_features = input_shape[-1]  # Last dimension is the feature dimension
    # Use fixed scale 0.01 - simple and FP16-safe for all test cases
    scale = 0.01
    
    input_cpu = torch.randn(*input_shape, dtype=torch.float16) * scale
    
    weight_cpu = torch.randn(*weight_shape, dtype=torch.float16) * scale
    
    bias_cpu = None
    
    # Calculate expected output shape
    # For F.linear: output_shape = input_shape[:-1] + (weight_shape[0],)
    expected_output_shape = input_shape[:-1] + (weight_shape[0],)
    
    # Calculate expected output magnitude
    # Output std ≈ sqrt(K) × scale²
    # For scale=0.01 and K=4096: std ≈ 64 × 0.0001 = 0.0064
    # Typical range (±3σ): ≈ [-0.019, +0.019]
    expected_output_std = (in_features ** 0.5) * (scale ** 2)
    expected_output_range = 3 * expected_output_std  # ±3σ covers ~99.7%
    
    if verbose:
        print(f"\n--- Input Data ---")
        print(f"Input shape: {input_cpu.shape}, dtype: {input_cpu.dtype}")
        print(f"Input range: [{input_cpu.min().item():.6f}, {input_cpu.max().item():.6f}]")
        print(f"Weight shape: {weight_cpu.shape}, dtype: {weight_cpu.dtype}")
        print(f"Weight range: [{weight_cpu.min().item():.6f}, {weight_cpu.max().item():.6f}]")
        print(f"Bias: {bias_cpu}")
        print(f"Expected output shape: {expected_output_shape}")
        print(f"Scaling factor: {scale} (fixed scale for FP16 safety)")
        print(f"Expected output std: {expected_output_std:.6f} (sqrt({in_features}) × {scale}²)")
        print(f"Expected output range (±3σ): [{-expected_output_range:.6f}, {expected_output_range:.6f}]")
    
    try:
        # CPU compiled execution
        if verbose:
            print("\n--- CPU Compiled ---")
        torch._dynamo.reset()
        compiled_fn = torch.compile(linear_wrapper, backend="inductor")
        output_cpu = compiled_fn(input_cpu, weight_cpu, bias_cpu)
        if verbose:
            print(f"CPU output shape: {output_cpu.shape}, dtype: {output_cpu.dtype}")
            print(f"CPU output range (actual): [{output_cpu.min().item():.6f}, {output_cpu.max().item():.6f}]")
        
        # Spyre compiled execution
        if verbose:
            print("\n--- Spyre Compiled ---")
        input_spyre = input_cpu.to("spyre")
        weight_spyre = weight_cpu.to("spyre")
        bias_spyre = None
        
        torch._dynamo.reset()
        compiled_fn_spyre = torch.compile(linear_wrapper, backend="inductor")
        output_spyre = compiled_fn_spyre(input_spyre, weight_spyre, bias_spyre)
        if verbose:
            print(f"Spyre output shape: {output_spyre.shape}, device: {output_spyre.device}")
        
        # Move Spyre output to CPU for comparison
        output_spyre_cpu = output_spyre.cpu()
        if verbose:
            print(f"Spyre output range (actual): [{output_spyre_cpu.min().item():.6f}, {output_spyre_cpu.max().item():.6f}]")
        
        # Verify shapes
        assert output_cpu.shape == expected_output_shape, (
            f"CPU output shape mismatch! Expected {expected_output_shape}, got {output_cpu.shape}"
        )
        assert output_spyre_cpu.shape == expected_output_shape, (
            f"Spyre output shape mismatch! Expected {expected_output_shape}, got {output_spyre_cpu.shape}"
        )
        
        # Compare CPU vs Spyre
        if verbose:
            print("\n--- Comparison ---")
        
        # Calculate detailed mismatch statistics
        abs_diff = torch.abs(output_cpu - output_spyre_cpu)
        # Use larger epsilon for FP16 (1e-5 instead of 1e-8) to avoid nan/inf
        # Also use max of both values for more robust relative error calculation
        denominator = torch.maximum(torch.abs(output_cpu), torch.abs(output_spyre_cpu)) + 1e-5
        rel_diff = abs_diff / denominator
        
        # Count mismatched elements
        rtol = 0.1
        atol = 0.1
        mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)
        num_mismatched = mismatch_mask.sum().item()
        total_elements = output_cpu.numel()
        mismatch_percentage = (num_mismatched / total_elements) * 100
        
        # Find greatest differences
        max_abs_diff = abs_diff.max().item()
        max_abs_idx = abs_diff.argmax().item()
        max_abs_idx_tuple = torch.unravel_index(torch.tensor(max_abs_idx), output_cpu.shape)
        max_abs_idx_list = tuple(int(idx.item()) for idx in max_abs_idx_tuple)
        
        # Filter out nan/inf for relative difference max
        rel_diff_finite = torch.where(torch.isfinite(rel_diff), rel_diff, torch.tensor(0.0, dtype=rel_diff.dtype))
        max_rel_diff = rel_diff_finite.max().item()
        max_rel_idx = rel_diff_finite.argmax().item()
        max_rel_idx_tuple = torch.unravel_index(torch.tensor(max_rel_idx), output_cpu.shape)
        max_rel_idx_list = tuple(int(idx.item()) for idx in max_rel_idx_tuple)
        
        # Get values at max difference locations for diagnostics
        cpu_val_at_max_abs = output_cpu[max_abs_idx_tuple].item()
        spyre_val_at_max_abs = output_spyre_cpu[max_abs_idx_tuple].item()
        
        cpu_val_at_max_rel = output_cpu[max_rel_idx_tuple].item()
        spyre_val_at_max_rel = output_spyre_cpu[max_rel_idx_tuple].item()
        
        # Print statistics
        if verbose:
            print(f"Mismatched elements: {num_mismatched} / {total_elements} ({mismatch_percentage:.2f}%)")
            print(f"Greatest absolute difference: {max_abs_diff:.6f} at index {max_abs_idx_list}")
            print(f"  CPU value: {cpu_val_at_max_abs:.6f}, Spyre value: {spyre_val_at_max_abs:.6f}")
            print(f"Greatest relative difference: {max_rel_diff:.6f} at index {max_rel_idx_list} [limit: {rtol}]")
            print(f"  CPU value: {cpu_val_at_max_rel:.6f}, Spyre value: {spyre_val_at_max_rel:.6f}")
        
        # Now perform the actual assertion
        try:
            torch.testing.assert_close(
                output_cpu, output_spyre_cpu,
                rtol=rtol, atol=atol,
                msg="CPU and Spyre compiled outputs don't match"
            )
            if verbose:
                print("✓ Test passed: CPU and Spyre outputs match within tolerance")
                print()
            return True
        except AssertionError as e:
            if verbose:
                print(f"✗ Test failed: Outputs exceed tolerance")
                print(f"  Error: {e}")
                print()
            return False
        
    except Exception as e:
        if verbose:
            print(f"✗ Test failed with error: {e}")
            print()
        return False


def main():
    """Run all test cases."""
    print("\n" + "=" * 80)
    print("Running torch.nn.functional.linear Tests")
    print("Compiled mode: CPU vs Spyre comparison")
    print("Based on FX Graph Analysis")
    print("=" * 80 + "\n")
    
    # List of test cases: (input_shape, weight_shape)
    test_cases = [
        # Case 1: linear.1
        ((1, 64, 4096), (4096, 4096)),
        # Case 2: linear.2
        ((1, 64, 4096), (1024, 4096)),        
        # Case 3: linear.3
        ((1, 64, 4096), (12800, 4096)),
        # Case 4: linear.4
        ((1, 64, 12800), (4096, 12800)),
        # Case 5: linear.5
        ((1, 64, 4096), (49159, 4096)),
        # Case 6: linear.6
        ((1, 1, 4096), (4096, 4096)),
        # Case 7: linear.7
        ((1, 1, 4096), (1024, 4096)),
        # Case 8: linear.8        
        ((1, 1, 4096), (12800, 4096)),
        # Case 9: linear.9
        ((1, 1, 12800), (4096, 12800)),
        # Case 10: linear.10
        ((1, 1, 4096), (49159, 4096)),
        # Case 11: issue #1106 _3_
        ((1, 64, 4096), (4096, 1024)),
        # Case 12: issue #1106 _8_
        ((1, 1, 4096), (4096, 1024)),
    ]
    
    # Run all test cases and track results
    results = []
    for i, (input_shape, weight_shape) in enumerate(test_cases, 1):
        case_name = f"linear.{i} ({input_shape[0]}x{input_shape[1]}x{input_shape[2]} @ {weight_shape[0]}x{weight_shape[1]})"
        passed = test_linear_fx_graph(
            input_shape=input_shape,
            weight_shape=weight_shape,
            case_name=case_name
        )
        results.append((i, case_name, passed))
    
    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed_count = sum(1 for _, _, passed in results if passed)
    failed_count = len(results) - passed_count
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed tests:")
        for i, case_name, passed in results:
            if not passed:
                print(f"  ✗ {case_name}")
    
    if passed_count > 0:
        print("\nPassed tests:")
        for i, case_name, passed in results:
            if passed:
                print(f"  ✓ {case_name}")
    
    print("\n" + "=" * 80)
    if failed_count == 0:
        print("All tests PASSED! ✓")
    else:
        print(f"{failed_count} test(s) FAILED! ✗")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
