#!/usr/bin/env python3
"""
Optimize Step Size for Perfect Coverage
=======================================

Calculate the optimal step size for 100% coverage without gaps.
"""

def find_optimal_step_size():
    """Find the perfect step size for 100% coverage"""

    ECG_SAMPLES = 2400
    CHUNK_SIZE = 140

    print("🎯 OPTIMIZING STEP SIZE FOR PERFECT 100% COVERAGE")
    print("=" * 60)

    # For perfect coverage with no gaps:
    # Last chunk should end exactly at sample 2400
    # So: (num_chunks - 1) * step_size + chunk_size = ECG_SAMPLES
    # Therefore: step_size = (ECG_SAMPLES - chunk_size) / (num_chunks - 1)

    print(f"\n📊 TARGET: Perfect coverage of {ECG_SAMPLES} samples")
    print(f"   Chunk size: {CHUNK_SIZE} samples")

    # Try different numbers of chunks
    options = []
    for num_chunks in range(14, 20):  # Around our target of 15
        step_size = (ECG_SAMPLES - CHUNK_SIZE) / (num_chunks - 1)
        step_size_int = int(step_size)

        # Calculate actual coverage with integer step size
        actual_end = (num_chunks - 1) * step_size_int + CHUNK_SIZE
        coverage_pct = (actual_end / ECG_SAMPLES) * 100

        # Check for gaps
        gaps = 0
        if step_size_int > CHUNK_SIZE:
            gaps = num_chunks - 1  # Number of gaps

        options.append({
            'chunks': num_chunks,
            'step_size_exact': step_size,
            'step_size_int': step_size_int,
            'coverage': coverage_pct,
            'end_sample': actual_end,
            'gaps': gaps,
            'comp_vs_current': (num_chunks / 10) * 100  # vs current 10 chunks
        })

    print(f"\n📋 STEP SIZE OPTIONS:")
    print(f"   {'Chunks':<7} {'Step Size':<10} {'Coverage':<10} {'End Sample':<11} {'Gaps':<5} {'Comp %'}")
    print("   " + "-" * 65)

    for opt in options:
        print(f"   {opt['chunks']:<7} {opt['step_size_int']:<10} {opt['coverage']:<9.1f}% {int(opt['end_sample']):<11} "
              f"{opt['gaps']:<5} {opt['comp_vs_current']:<6.0f}%")

    # Find best option (high coverage, reasonable computational cost)
    best = min([opt for opt in options if opt['coverage'] >= 99.5],
               key=lambda x: x['comp_vs_current'])

    print(f"\n🏆 RECOMMENDED OPTIMAL STEP SIZE:")
    print(f"   Chunks per lead: {best['chunks']}")
    print(f"   Step size: {best['step_size_int']} samples")
    print(f"   Coverage: {best['coverage']:.1f}%")
    print(f"   End sample: {int(best['end_sample'])} (target: {ECG_SAMPLES})")
    print(f"   Computational load: {best['comp_vs_current']:.0f}% of current")

    # Show exact chunk positions with optimal step size
    step_size = best['step_size_int']
    print(f"\n📍 CHUNK POSITIONS WITH STEP SIZE {step_size}:")

    chunks = []
    for start in range(0, ECG_SAMPLES - CHUNK_SIZE + 1, step_size):
        end = start + CHUNK_SIZE
        if end <= ECG_SAMPLES:
            chunks.append((start, end))

    print(f"   {'Chunk':<6} {'Start':<6} {'End':<6} {'Gap After'}")
    print("   " + "-" * 35)

    for i, (start, end) in enumerate(chunks):
        gap = ""
        if i < len(chunks) - 1:
            next_start = chunks[i + 1][0]
            if next_start > end:
                gap = f"Gap: {end}-{next_start}"
            elif next_start == end:
                gap = "No gap"
            else:
                gap = f"Overlap: {end - next_start}"

        print(f"   {i+1:<6} {start:<6} {end:<6} {gap}")

    return best['step_size_int']

if __name__ == "__main__":
    optimal_step = find_optimal_step_size()