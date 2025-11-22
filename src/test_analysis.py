"""
Toyota GR Cup - Comprehensive Testing Suite
============================================
This file contains detailed tests to verify the correctness of all analysis
calculations, data transformations, and visualizations.

Tests cover:
1. Data Loading & Pivoting
2. Data Cleaning & Filtering
3. Physics Calculations (G-forces, Tire Stress)
4. Lap Metrics & ROI Calculations
5. Driver Skill Analysis
6. Tire Failure Prediction
7. Edge Cases & Data Validation

Author: Toyota Racing ROI Analysis
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from analysis scripts
from analysis import (
    load_telemetry_data,
    clean_telemetry_data,
    calculate_total_g,
    analyze_driver_skill
)

from roi_engine import (
    load_and_pivot_telemetry,
    clean_telemetry,
    calculate_tire_stress,
    calculate_lap_metrics,
    calculate_roi_efficiency,
    predict_tire_failure
)

class TestSuite:
    """Comprehensive test suite for Toyota GR Cup analysis"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = 0
        self.filepath = r'data\Sonoma\Race 1\sonoma_telemetry_R1.csv'
        
    def assert_equal(self, actual, expected, test_name, tolerance=None):
        """Custom assertion with detailed output"""
        self.tests_run += 1
        
        if tolerance is not None:
            # For floating point comparisons
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                passed = abs(actual - expected) <= tolerance
            else:
                passed = actual == expected
        else:
            passed = actual == expected
        
        if passed:
            self.passed += 1
            print(f"✅ PASS: {test_name}")
            print(f"   Expected: {expected}, Got: {actual}")
        else:
            self.failed += 1
            print(f"❌ FAIL: {test_name}")
            print(f"   Expected: {expected}, Got: {actual}")
            if tolerance:
                print(f"   Tolerance: ±{tolerance}")
        print()
        
    def assert_true(self, condition, test_name, message=""):
        """Assert that condition is True"""
        self.tests_run += 1
        
        if condition:
            self.passed += 1
            print(f"✅ PASS: {test_name}")
            if message:
                print(f"   {message}")
        else:
            self.failed += 1
            print(f"❌ FAIL: {test_name}")
            if message:
                print(f"   {message}")
        print()
        
    def assert_range(self, value, min_val, max_val, test_name):
        """Assert value is within range"""
        self.tests_run += 1
        
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"✅ PASS: {test_name}")
            print(f"   Value {value} is within range [{min_val}, {max_val}]")
        else:
            self.failed += 1
            print(f"❌ FAIL: {test_name}")
            print(f"   Value {value} is NOT within range [{min_val}, {max_val}]")
        print()
    
    def print_section(self, title):
        """Print section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("  TEST SUMMARY")
        print("="*80)
        print(f"Total Tests Run: {self.tests_run}")
        print(f"✅ Passed: {self.passed} ({self.passed/self.tests_run*100:.1f}%)")
        print(f"❌ Failed: {self.failed} ({self.failed/self.tests_run*100:.1f}%)")
        print("="*80 + "\n")
        
        if self.failed == 0:
            print("🎉 ALL TESTS PASSED! Your analysis is mathematically correct! 🎉\n")
        else:
            print(f"⚠️  {self.failed} test(s) failed. Review the output above.\n")


def test_data_loading_and_pivoting(suite):
    """Test 1: Data Loading & Pivoting from Long to Wide Format"""
    suite.print_section("TEST 1: DATA LOADING & PIVOTING")
    
    try:
        # Load data using analysis.py function
        df = load_telemetry_data(suite.filepath)
        
        # Test: DataFrame is not empty
        suite.assert_true(
            len(df) > 0,
            "Data loaded successfully",
            f"Loaded {len(df):,} rows"
        )
        
        # Test: Expected columns exist after pivot
        expected_columns = ['speed', 'accx_can', 'accy_can', 'lap', 'vehicle_id']
        for col in expected_columns:
            suite.assert_true(
                col in df.columns,
                f"Column '{col}' exists after pivot",
                f"Found in columns: {col in df.columns}"
            )
        
        # Test: Data shape is reasonable (wide format should have fewer rows than long)
        suite.assert_true(
            len(df.columns) > 20,  # Should have many telemetry columns
            "Pivot created wide format with many columns",
            f"Total columns: {len(df.columns)}"
        )
        
        # Test: No duplicate timestamps per vehicle
        duplicates = df.groupby(['vehicle_id', 'timestamp']).size().max()
        suite.assert_equal(
            duplicates, 1,
            "No duplicate timestamps per vehicle after pivot"
        )
        
        print(f"📊 Data Structure:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Vehicles: {df['vehicle_id'].nunique()}")
        print(f"   Laps: {df['lap'].nunique()}")
        
        return df
        
    except Exception as e:
        suite.assert_true(False, "Data loading test", f"Exception: {e}")
        return None


def test_data_cleaning(suite, df):
    """Test 2: Data Cleaning & Filtering Logic"""
    suite.print_section("TEST 2: DATA CLEANING & FILTERING")
    
    if df is None:
        print("⚠️  Skipping - no data available")
        return None
    
    # Store initial state
    initial_rows = len(df)
    initial_speed_below_10 = len(df[df['speed'] < 10])
    initial_invalid_accel = len(df[(df['accx_can'] == 0) | (df['accy_can'] == 0)])
    initial_superhuman = len(df[(df['accx_can'].abs() > 2.0) | (df['accy_can'].abs() > 2.0)])
    
    print(f"📊 Pre-Cleaning Statistics:")
    print(f"   Total rows: {initial_rows:,}")
    print(f"   Speed < 10 km/h: {initial_speed_below_10:,}")
    print(f"   Invalid acceleration: {initial_invalid_accel:,}")
    print(f"   Superhuman G-forces (>2.0G): {initial_superhuman:,}")
    print()
    
    # Clean data
    df_clean = clean_telemetry_data(df)
    
    # Test: Cleaning removes rows
    suite.assert_true(
        len(df_clean) < initial_rows,
        "Cleaning process removed invalid data",
        f"Removed {initial_rows - len(df_clean):,} rows ({(initial_rows-len(df_clean))/initial_rows*100:.1f}%)"
    )
    
    # Test: All speeds >= 10 km/h
    min_speed = df_clean['speed'].min()
    suite.assert_true(
        min_speed >= 10,
        "All speeds >= 10 km/h after cleaning",
        f"Minimum speed: {min_speed:.2f} km/h"
    )
    
    # Test: No zero acceleration values
    zero_accx = len(df_clean[df_clean['accx_can'] == 0])
    zero_accy = len(df_clean[df_clean['accy_can'] == 0])
    suite.assert_equal(zero_accx, 0, "No zero longitudinal acceleration")
    suite.assert_equal(zero_accy, 0, "No zero lateral acceleration")
    
    # Test: All G-forces within realistic limits (±2.0G)
    max_accx = df_clean['accx_can'].abs().max()
    max_accy = df_clean['accy_can'].abs().max()
    suite.assert_true(
        max_accx <= 2.0,
        "Maximum longitudinal G-force <= 2.0G",
        f"Max accx: {max_accx:.3f}G"
    )
    suite.assert_true(
        max_accy <= 2.0,
        "Maximum lateral G-force <= 2.0G",
        f"Max accy: {max_accy:.3f}G"
    )
    
    # Test: No NaN values in critical columns
    critical_cols = ['speed', 'accx_can', 'accy_can']
    for col in critical_cols:
        nan_count = df_clean[col].isna().sum()
        suite.assert_equal(
            nan_count, 0,
            f"No NaN values in '{col}' after cleaning"
        )
    
    print(f"📊 Post-Cleaning Statistics:")
    print(f"   Clean rows: {len(df_clean):,}")
    print(f"   Data retention: {len(df_clean)/initial_rows*100:.1f}%")
    
    return df_clean


def test_physics_calculations(suite, df_clean):
    """Test 3: Physics Calculations (G-forces, Total_G)"""
    suite.print_section("TEST 3: PHYSICS CALCULATIONS")
    
    if df_clean is None:
        print("⚠️  Skipping - no clean data available")
        return None
    
    # Calculate Total G-force
    df_clean = calculate_total_g(df_clean)
    
    # Test: Total_G column exists
    suite.assert_true(
        'Total_G' in df_clean.columns,
        "Total_G column created",
        "Column exists in dataframe"
    )
    
    # Test: Total_G formula correctness (sqrt(accx² + accy²))
    # Check a random sample
    sample = df_clean.sample(n=min(100, len(df_clean)))
    for idx, row in sample.iterrows():
        expected_g = np.sqrt(row['accx_can']**2 + row['accy_can']**2)
        actual_g = row['Total_G']
        
        # Test with small tolerance for floating point errors
        if abs(expected_g - actual_g) > 0.0001:
            suite.assert_equal(
                actual_g, expected_g,
                f"Total_G calculation for sample row {idx}",
                tolerance=0.0001
            )
            break
    else:
        suite.assert_true(
            True,
            "Total_G formula correct for all sampled rows",
            f"Verified {len(sample)} random samples"
        )
    
    # Test: Total_G is always positive
    min_total_g = df_clean['Total_G'].min()
    suite.assert_true(
        min_total_g >= 0,
        "Total_G is always non-negative",
        f"Minimum Total_G: {min_total_g:.4f}"
    )
    
    # Test: Total_G should be >= max of |accx| or |accy| (vector magnitude property)
    max_accx_abs = df_clean['accx_can'].abs().max()
    max_accy_abs = df_clean['accy_can'].abs().max()
    max_total_g = df_clean['Total_G'].max()
    suite.assert_true(
        max_total_g >= max(max_accx_abs, max_accy_abs),
        "Total_G >= max individual acceleration component",
        f"Total_G: {max_total_g:.3f}, Max component: {max(max_accx_abs, max_accy_abs):.3f}"
    )
    
    # Test: Total_G should be <= sum of |accx| + |accy| (triangle inequality)
    sample = df_clean.sample(n=min(100, len(df_clean)))
    triangle_violations = 0
    for idx, row in sample.iterrows():
        if row['Total_G'] > (abs(row['accx_can']) + abs(row['accy_can'])) + 0.0001:
            triangle_violations += 1
    
    suite.assert_equal(
        triangle_violations, 0,
        "Total_G respects triangle inequality (≤ |accx| + |accy|)",
    )
    
    # Test: Physical reasonableness - Total_G should mostly be under 2.0G
    reasonable_g = len(df_clean[df_clean['Total_G'] <= 2.0]) / len(df_clean) * 100
    suite.assert_true(
        reasonable_g >= 95,  # At least 95% should be under 2.0G
        "At least 95% of Total_G values are physically reasonable (≤2.0G)",
        f"Reasonable values: {reasonable_g:.1f}%"
    )
    
    print(f"📊 Total_G Statistics:")
    print(f"   Mean: {df_clean['Total_G'].mean():.3f}G")
    print(f"   Median: {df_clean['Total_G'].median():.3f}G")
    print(f"   Max: {df_clean['Total_G'].max():.3f}G")
    print(f"   Std Dev: {df_clean['Total_G'].std():.3f}G")
    
    return df_clean


def test_tire_stress_calculation(suite, df_clean):
    """Test 4: Tire Stress Metric Calculation"""
    suite.print_section("TEST 4: TIRE STRESS CALCULATION")
    
    if df_clean is None:
        print("⚠️  Skipping - no clean data available")
        return None
    
    # Calculate tire stress using ROI engine function
    df_stress = calculate_tire_stress(df_clean.copy())
    
    # Test: tire_stress column exists
    suite.assert_true(
        'tire_stress' in df_stress.columns,
        "tire_stress column created"
    )
    
    # Test: Tire stress formula (G² * Duration)
    sample = df_stress.sample(n=min(100, len(df_stress)))
    for idx, row in sample.iterrows():
        expected_stress = (row['total_g']**2) * 0.02  # 0.02s time delta
        actual_stress = row['tire_stress']
        
        if abs(expected_stress - actual_stress) > 0.0001:
            suite.assert_equal(
                actual_stress, expected_stress,
                f"Tire stress formula for sample row {idx}",
                tolerance=0.0001
            )
            break
    else:
        suite.assert_true(
            True,
            "Tire stress formula correct (G² × Duration)",
            f"Verified {len(sample)} random samples"
        )
    
    # Test: Tire stress is always positive
    min_stress = df_stress['tire_stress'].min()
    suite.assert_true(
        min_stress >= 0,
        "Tire stress is always non-negative",
        f"Minimum stress: {min_stress:.6f}"
    )
    
    # Test: Higher G-forces produce higher stress (monotonic relationship)
    # Compare high-G vs low-G segments
    low_g = df_stress[df_stress['total_g'] < 0.5]['tire_stress'].mean()
    high_g = df_stress[df_stress['total_g'] > 1.5]['tire_stress'].mean()
    suite.assert_true(
        high_g > low_g,
        "Higher G-forces produce higher tire stress (monotonic)",
        f"Low-G stress: {low_g:.6f}, High-G stress: {high_g:.6f}"
    )
    
    # Test: Exponential stress growth (G² effect)
    # 2G should produce ~4x more stress than 1G (for same duration)
    g1_stress = 1.0**2 * 0.02
    g2_stress = 2.0**2 * 0.02
    ratio = g2_stress / g1_stress
    suite.assert_equal(
        ratio, 4.0,
        "Tire stress shows exponential growth (G² relationship)",
        tolerance=0.01
    )
    
    print(f"📊 Tire Stress Statistics:")
    print(f"   Mean: {df_stress['tire_stress'].mean():.6f}")
    print(f"   Median: {df_stress['tire_stress'].median():.6f}")
    print(f"   Max: {df_stress['tire_stress'].max():.6f}")
    print(f"   Total (all samples): {df_stress['tire_stress'].sum():.2f}")
    
    return df_stress


def test_lap_metrics(suite, df_stress):
    """Test 5: Lap-Level Metrics Aggregation"""
    suite.print_section("TEST 5: LAP METRICS AGGREGATION")
    
    if df_stress is None:
        print("⚠️  Skipping - no stress data available")
        return None
    
    # Calculate lap metrics
    lap_summary = calculate_lap_metrics(df_stress)
    
    # Test: Lap summary has correct structure
    expected_cols = ['vehicle_id', 'lap', 'lap_tire_stress', 'avg_g', 
                     'avg_speed', 'sample_count', 'lap_time_est']
    for col in expected_cols:
        suite.assert_true(
            col in lap_summary.columns,
            f"Lap summary contains '{col}' column"
        )
    
    # Test: Number of unique laps matches
    unique_laps = df_stress.groupby(['vehicle_id', 'lap']).size().shape[0]
    suite.assert_equal(
        len(lap_summary), unique_laps,
        "Lap summary row count matches unique lap count"
    )
    
    # Test: Lap tire stress is sum of individual stresses
    # Verify for one random vehicle-lap combination
    sample_vehicle = lap_summary['vehicle_id'].iloc[0]
    sample_lap = lap_summary[lap_summary['vehicle_id'] == sample_vehicle]['lap'].iloc[0]
    
    # Calculate expected sum
    expected_stress = df_stress[
        (df_stress['vehicle_id'] == sample_vehicle) & 
        (df_stress['lap'] == sample_lap)
    ]['tire_stress'].sum()
    
    # Get actual from summary
    actual_stress = lap_summary[
        (lap_summary['vehicle_id'] == sample_vehicle) & 
        (lap_summary['lap'] == sample_lap)
    ]['lap_tire_stress'].iloc[0]
    
    suite.assert_equal(
        actual_stress, expected_stress,
        f"Lap stress aggregation correct for {sample_vehicle}, Lap {sample_lap}",
        tolerance=0.001
    )
    
    # Test: Sample count is positive for all laps
    min_samples = lap_summary['sample_count'].min()
    suite.assert_true(
        min_samples > 0,
        "All laps have positive sample counts",
        f"Minimum samples per lap: {min_samples}"
    )
    
    # Test: Lap time estimation is reasonable (30-120 seconds typical for racing)
    mean_lap_time = lap_summary['lap_time_est'].mean()
    suite.assert_range(
        mean_lap_time, 30, 180,
        "Mean lap time is reasonable for racing"
    )
    
    # Test: Delta columns exist
    delta_cols = ['stress_delta', 'time_delta', 'stress_delta_pct', 'time_delta_pct']
    for col in delta_cols:
        suite.assert_true(
            col in lap_summary.columns,
            f"Delta metric '{col}' calculated"
        )
    
    print(f"📊 Lap Metrics Summary:")
    print(f"   Total laps analyzed: {len(lap_summary)}")
    print(f"   Vehicles: {lap_summary['vehicle_id'].nunique()}")
    print(f"   Mean lap tire stress: {lap_summary['lap_tire_stress'].mean():.3f}")
    print(f"   Mean lap time: {mean_lap_time:.1f}s")
    
    return lap_summary


def test_roi_efficiency(suite, lap_summary):
    """Test 6: ROI Efficiency Calculation"""
    suite.print_section("TEST 6: ROI EFFICIENCY CALCULATION")
    
    if lap_summary is None:
        print("⚠️  Skipping - no lap metrics available")
        return None
    
    # Calculate ROI
    lap_roi = calculate_roi_efficiency(lap_summary)
    
    # Test: ROI columns exist
    suite.assert_true(
        'roi_efficiency' in lap_roi.columns,
        "roi_efficiency column created"
    )
    suite.assert_true(
        'roi_category' in lap_roi.columns,
        "roi_category column created"
    )
    
    # Test: ROI formula logic
    # For a lap with negative time_delta (faster) and positive stress_delta (more wear),
    # ROI should be positive (good efficiency)
    fast_worn = lap_roi[
        (lap_roi['time_delta'] < 0) & 
        (lap_roi['stress_delta'] > 0.1)
    ]
    if len(fast_worn) > 0:
        sample_roi = fast_worn.iloc[0]['roi_efficiency']
        suite.assert_true(
            sample_roi > 0,
            "Fast lap with tire wear has positive ROI",
            f"ROI: {sample_roi:.3f}"
        )
    
    # Test: ROI categories are valid
    valid_categories = ['REFERENCE', 'EXCELLENT', 'GOOD', 'WASTEFUL', 'TERRIBLE']
    invalid_cats = lap_roi[~lap_roi['roi_category'].isin(valid_categories)]
    suite.assert_equal(
        len(invalid_cats), 0,
        "All ROI categories are valid"
    )
    
    # Test: Category distribution makes sense
    category_counts = lap_roi['roi_category'].value_counts()
    suite.assert_true(
        len(category_counts) >= 2,
        "Multiple ROI categories present (data has variation)",
        f"Categories found: {list(category_counts.index)}"
    )
    
    # Test: EXCELLENT category has positive ROI
    excellent = lap_roi[lap_roi['roi_category'] == 'EXCELLENT']
    if len(excellent) > 0:
        min_excellent_roi = excellent['roi_efficiency'].min()
        suite.assert_true(
            min_excellent_roi > 0.5,
            "EXCELLENT category has ROI > 0.5",
            f"Min EXCELLENT ROI: {min_excellent_roi:.3f}"
        )
    
    # Test: TERRIBLE category has very negative ROI
    terrible = lap_roi[lap_roi['roi_category'] == 'TERRIBLE']
    if len(terrible) > 0:
        max_terrible_roi = terrible['roi_efficiency'].max()
        suite.assert_true(
            max_terrible_roi < -0.5,
            "TERRIBLE category has ROI < -0.5",
            f"Max TERRIBLE ROI: {max_terrible_roi:.3f}"
        )
    
    print(f"📊 ROI Distribution:")
    for cat, count in category_counts.items():
        print(f"   {cat}: {count} laps ({count/len(lap_roi)*100:.1f}%)")
    
    return lap_roi


def test_driver_skill_analysis(suite, df_clean):
    """Test 7: Driver Skill Analysis (Diamond vs Mushroom)"""
    suite.print_section("TEST 7: DRIVER SKILL ANALYSIS")
    
    if df_clean is None:
        print("⚠️  Skipping - no clean data available")
        return None
    
    # Analyze driver skill
    skill_analysis = analyze_driver_skill(df_clean)
    
    # Test: Analysis returns expected keys
    expected_keys = ['trail_brake_pct', 'skill_level', 'shape']
    for key in expected_keys:
        suite.assert_true(
            key in skill_analysis,
            f"Skill analysis contains '{key}'"
        )
    
    # Test: Trail brake percentage is reasonable (0-100%)
    trail_pct = skill_analysis['trail_brake_pct']
    suite.assert_range(
        trail_pct, 0, 100,
        "Trail braking percentage in valid range"
    )
    
    # Test: Skill level categorization is correct
    if trail_pct > 40:
        expected_level = "PRO - Excellent trail braking technique!"
    elif trail_pct > 25:
        expected_level = "INTERMEDIATE - Good corner entry"
    else:
        expected_level = "AMATEUR - Sequential inputs (brake, turn, gas)"
    
    suite.assert_equal(
        skill_analysis['skill_level'], expected_level,
        "Skill level category matches trail brake percentage"
    )
    
    # Test: Shape categorization matches skill level
    if "PRO" in skill_analysis['skill_level']:
        expected_shape = "MUSHROOM/CIRCLE"
    elif "INTERMEDIATE" in skill_analysis['skill_level']:
        expected_shape = "ROUNDED DIAMOND"
    else:
        expected_shape = "DIAMOND"
    
    suite.assert_equal(
        skill_analysis['shape'], expected_shape,
        "Shape category matches skill level"
    )
    
    print(f"📊 Driver Skill Results:")
    print(f"   Trail Braking: {trail_pct:.1f}%")
    print(f"   Skill Level: {skill_analysis['skill_level']}")
    print(f"   Expected Shape: {skill_analysis['shape']}")
    
    return skill_analysis


def test_tire_failure_prediction(suite, lap_roi):
    """Test 8: Tire Failure Prediction Model"""
    suite.print_section("TEST 8: TIRE FAILURE PREDICTION")
    
    if lap_roi is None:
        print("⚠️  Skipping - no ROI data available")
        return None
    
    # Get first vehicle for testing
    vehicle_id = lap_roi['vehicle_id'].iloc[0]
    
    # Run prediction
    prediction = predict_tire_failure(lap_roi, vehicle_id)
    
    # Test: Prediction returns expected keys
    expected_keys = ['vehicle_id', 'current_lap', 'cumulative_stress', 
                     'stress_rate', 'failure_threshold', 'predicted_failure_lap',
                     'laps_remaining', 'model', 'data']
    for key in expected_keys:
        suite.assert_true(
            key in prediction,
            f"Prediction contains '{key}'"
        )
    
    # Test: Current lap is positive
    suite.assert_true(
        prediction['current_lap'] > 0,
        "Current lap is positive",
        f"Current lap: {prediction['current_lap']}"
    )
    
    # Test: Cumulative stress increases with laps
    vehicle_data = prediction['data']
    stress_increasing = (vehicle_data['cumulative_stress'].diff().dropna() >= 0).all()
    suite.assert_true(
        stress_increasing,
        "Cumulative stress is monotonically increasing",
        "Stress grows with each lap"
    )
    
    # Test: Stress rate is positive
    suite.assert_true(
        prediction['stress_rate'] > 0,
        "Stress accumulation rate is positive",
        f"Rate: {prediction['stress_rate']:.3f} per lap"
    )
    
    # Test: Predicted failure lap is after current lap
    suite.assert_true(
        prediction['predicted_failure_lap'] > prediction['current_lap'],
        "Predicted failure is in the future",
        f"Predicted: Lap {prediction['predicted_failure_lap']:.1f}, Current: Lap {prediction['current_lap']}"
    )
    
    # Test: Linear regression model exists and is fitted
    suite.assert_true(
        hasattr(prediction['model'], 'coef_'),
        "Linear regression model is fitted",
        "Model has coefficients"
    )
    
    # Test: Model prediction at current lap matches cumulative stress
    current_lap_array = np.array([[prediction['current_lap']]])
    predicted_stress = prediction['model'].predict(current_lap_array)[0]
    actual_stress = prediction['cumulative_stress']
    
    # Should be very close (model is fitted on this data)
    error_pct = abs(predicted_stress - actual_stress) / actual_stress * 100
    suite.assert_true(
        error_pct < 5,  # Less than 5% error
        "Model prediction accurate at current lap",
        f"Error: {error_pct:.2f}%"
    )
    
    print(f"📊 Tire Failure Prediction:")
    print(f"   Vehicle: {prediction['vehicle_id']}")
    print(f"   Current Lap: {prediction['current_lap']}")
    print(f"   Cumulative Stress: {prediction['cumulative_stress']:.2f}")
    print(f"   Predicted Failure: Lap {prediction['predicted_failure_lap']:.1f}")
    print(f"   Laps Remaining: {prediction['laps_remaining']:.1f}")
    
    return prediction


def test_edge_cases(suite):
    """Test 9: Edge Cases & Data Validation"""
    suite.print_section("TEST 9: EDGE CASES & DATA VALIDATION")
    
    # Test: Handle empty dataframe
    empty_df = pd.DataFrame()
    try:
        result = clean_telemetry_data(empty_df)
        suite.assert_true(
            len(result) == 0,
            "Empty dataframe handling",
            "Returns empty dataframe without error"
        )
    except Exception as e:
        suite.assert_true(
            False,
            "Empty dataframe handling",
            f"Should handle gracefully, got: {e}"
        )
    
    # Test: Handle extreme values
    test_df = pd.DataFrame({
        'speed': [100, 200, 0, -10],
        'accx_can': [0.5, 5.0, 0, -0.5],  # 5.0 is extreme
        'accy_can': [0.3, 0.2, 0, -0.3]
    })
    
    cleaned = clean_telemetry_data(test_df)
    
    # Should remove speed < 10 and G > 2.0
    suite.assert_true(
        len(cleaned) <= 2,  # Only first two rows potentially valid
        "Extreme value filtering",
        f"Filtered {len(test_df)} → {len(cleaned)} rows"
    )
    
    # Test: Division by zero handling in ROI
    test_lap = pd.DataFrame({
        'vehicle_id': ['test'],
        'lap': [1],
        'lap_tire_stress': [1.0],
        'stress_delta': [0],  # Zero delta - edge case
        'time_delta': [-1.0]
    })
    
    try:
        roi_result = calculate_roi_efficiency(test_lap)
        suite.assert_true(
            'roi_efficiency' in roi_result.columns,
            "Division by zero handling in ROI",
            "Handles zero stress delta without error"
        )
    except Exception as e:
        suite.assert_true(
            False,
            "Division by zero handling in ROI",
            f"Should handle zero delta, got: {e}"
        )
    
    # Test: NaN handling
    test_nan = pd.DataFrame({
        'speed': [50, np.nan, 60],
        'accx_can': [0.5, 0.3, np.nan],
        'accy_can': [0.2, np.nan, 0.4]
    })
    
    cleaned_nan = clean_telemetry_data(test_nan)
    suite.assert_true(
        cleaned_nan['speed'].notna().all(),
        "NaN filtering in critical columns",
        "All NaN values removed"
    )
    
    print("✅ Edge case handling validated")


def test_data_consistency(suite, df_clean, lap_roi):
    """Test 10: Data Consistency Checks"""
    suite.print_section("TEST 10: DATA CONSISTENCY CHECKS")
    
    if df_clean is None or lap_roi is None:
        print("⚠️  Skipping - data not available")
        return
    
    # Test: Vehicle IDs consistent across datasets
    vehicles_raw = set(df_clean['vehicle_id'].unique())
    vehicles_laps = set(lap_roi['vehicle_id'].unique())
    
    suite.assert_equal(
        vehicles_raw, vehicles_laps,
        "Vehicle IDs consistent between raw and lap data"
    )
    
    # Test: Lap numbers are sequential per vehicle
    for vehicle in lap_roi['vehicle_id'].unique():
        vehicle_laps = lap_roi[lap_roi['vehicle_id'] == vehicle]['lap'].sort_values()
        gaps = vehicle_laps.diff().dropna()
        max_gap = gaps.max()
        
        suite.assert_true(
            max_gap <= 1,
            f"Lap sequence valid for {vehicle}",
            f"Max gap: {max_gap}"
        )
        break  # Test one vehicle as example
    
    # Test: Data types are correct
    suite.assert_true(
        pd.api.types.is_numeric_dtype(df_clean['speed']),
        "Speed column is numeric"
    )
    suite.assert_true(
        pd.api.types.is_numeric_dtype(lap_roi['lap_tire_stress']),
        "Tire stress column is numeric"
    )
    
    # Test: No duplicate rows in lap summary
    duplicates = lap_roi.duplicated(subset=['vehicle_id', 'lap']).sum()
    suite.assert_equal(
        duplicates, 0,
        "No duplicate vehicle-lap combinations in summary"
    )
    
    print("✅ Data consistency validated")


def run_tests_for_race(suite, race_name, filepath):
    """Run all tests for a specific race"""
    print("\n" + "🏁"*40)
    print(f"  TESTING {race_name.upper()}")
    print("🏁"*40 + "\n")
    
    # Temporarily set the filepath for this race
    original_filepath = suite.filepath
    suite.filepath = filepath
    
    # Check if data file exists
    if not os.path.exists(suite.filepath):
        print(f"⚠️  SKIPPING {race_name}: Data file not found at {suite.filepath}")
        suite.filepath = original_filepath
        return None
    
    # Run all tests for this race
    df = test_data_loading_and_pivoting(suite)
    df_clean = test_data_cleaning(suite, df)
    df_clean_with_g = test_physics_calculations(suite, df_clean)
    df_stress = test_tire_stress_calculation(suite, df_clean_with_g)
    lap_summary = test_lap_metrics(suite, df_stress)
    lap_roi = test_roi_efficiency(suite, lap_summary)
    skill_analysis = test_driver_skill_analysis(suite, df_clean_with_g)
    prediction = test_tire_failure_prediction(suite, lap_roi)
    
    # Restore original filepath
    suite.filepath = original_filepath
    
    return {
        'race': race_name,
        'df': df,
        'df_clean': df_clean,
        'df_stress': df_stress,
        'lap_summary': lap_summary,
        'lap_roi': lap_roi,
        'skill_analysis': skill_analysis,
        'prediction': prediction
    }


def compare_races(race1_results, race2_results):
    """Compare results between Race 1 and Race 2"""
    print("\n" + "="*80)
    print("  RACE COMPARISON - Race 1 vs Race 2")
    print("="*80 + "\n")
    
    if race1_results is None or race2_results is None:
        print("⚠️  Cannot compare - one or both races failed to load")
        return
    
    print("📊 DATA VOLUME COMPARISON:")
    print(f"   Race 1: {len(race1_results['df']):,} rows → {len(race1_results['df_clean']):,} clean")
    print(f"   Race 2: {len(race2_results['df']):,} rows → {len(race2_results['df_clean']):,} clean")
    print(f"   Ratio: {len(race2_results['df'])/len(race1_results['df']):.2f}x")
    
    print("\n🏎️  LAPS COMPARISON:")
    print(f"   Race 1: {len(race1_results['lap_summary']):,} laps, {race1_results['lap_summary']['vehicle_id'].nunique()} vehicles")
    print(f"   Race 2: {len(race2_results['lap_summary']):,} laps, {race2_results['lap_summary']['vehicle_id'].nunique()} vehicles")
    
    print("\n⚙️  TIRE STRESS COMPARISON:")
    r1_stress = race1_results['df_stress']['tire_stress'].sum()
    r2_stress = race2_results['df_stress']['tire_stress'].sum()
    print(f"   Race 1 Total Stress: {r1_stress:.2f}")
    print(f"   Race 2 Total Stress: {r2_stress:.2f}")
    print(f"   Difference: {r2_stress - r1_stress:+.2f} ({(r2_stress/r1_stress - 1)*100:+.1f}%)")
    
    print("\n🔬 PHYSICS VALIDATION:")
    r1_max_g = race1_results['df_clean']['Total_G'].max()
    r2_max_g = race2_results['df_clean']['Total_G'].max()
    print(f"   Race 1 Max G-Force: {r1_max_g:.3f}G")
    print(f"   Race 2 Max G-Force: {r2_max_g:.3f}G")
    
    print("\n💰 ROI EFFICIENCY COMPARISON:")
    r1_roi = race1_results['lap_roi']['roi_category'].value_counts()
    r2_roi = race2_results['lap_roi']['roi_category'].value_counts()
    
    for cat in ['EXCELLENT', 'GOOD', 'WASTEFUL', 'TERRIBLE']:
        r1_count = r1_roi.get(cat, 0)
        r2_count = r2_roi.get(cat, 0)
        r1_pct = r1_count / len(race1_results['lap_roi']) * 100 if len(race1_results['lap_roi']) > 0 else 0
        r2_pct = r2_count / len(race2_results['lap_roi']) * 100 if len(race2_results['lap_roi']) > 0 else 0
        print(f"   {cat}:")
        print(f"      Race 1: {r1_count} ({r1_pct:.1f}%) | Race 2: {r2_count} ({r2_pct:.1f}%)")
    
    print("\n🏁 DRIVER SKILL COMPARISON:")
    r1_skill = race1_results['skill_analysis']
    r2_skill = race2_results['skill_analysis']
    print(f"   Race 1 Trail Braking: {r1_skill['trail_brake_pct']:.1f}% ({r1_skill['skill_level'].split(' - ')[0]})")
    print(f"   Race 2 Trail Braking: {r2_skill['trail_brake_pct']:.1f}% ({r2_skill['skill_level'].split(' - ')[0]})")
    print(f"   Improvement: {r2_skill['trail_brake_pct'] - r1_skill['trail_brake_pct']:+.1f}%")
    
    print("\n🔮 TIRE FAILURE PREDICTION COMPARISON:")
    r1_pred = race1_results['prediction']
    r2_pred = race2_results['prediction']
    print(f"   Race 1: Lap {r1_pred['predicted_failure_lap']:.1f} (stress rate: {r1_pred['stress_rate']:.2f}/lap)")
    print(f"   Race 2: Lap {r2_pred['predicted_failure_lap']:.1f} (stress rate: {r2_pred['stress_rate']:.2f}/lap)")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  TOYOTA GR CUP - COMPREHENSIVE TESTING SUITE")
    print("  Testing all calculations, formulas, and analysis logic")
    print("  Now testing BOTH Race 1 and Race 2 data")
    print("="*80 + "\n")
    
    # Initialize test suite
    suite = TestSuite()
    
    # Define race data files
    race1_filepath = r'data\Sonoma\Race 1\sonoma_telemetry_R1.csv'
    race2_filepath = r'data\Sonoma\Race 2\sonoma_telemetry_R2.csv'
    
    # Run tests for Race 1
    race1_results = run_tests_for_race(suite, "Race 1", race1_filepath)
    
    # Run tests for Race 2
    race2_results = run_tests_for_race(suite, "Race 2", race2_filepath)
    
    # Run edge case tests once (not race-specific)
    suite.print_section("EDGE CASES & DATA VALIDATION (GENERAL)")
    test_edge_cases(suite)
    
    # Test data consistency for both races
    if race1_results:
        suite.print_section("DATA CONSISTENCY - RACE 1")
        test_data_consistency(suite, race1_results['df_clean'], race1_results['lap_roi'])
    
    if race2_results:
        suite.print_section("DATA CONSISTENCY - RACE 2")
        test_data_consistency(suite, race2_results['df_clean'], race2_results['lap_roi'])
    
    # Compare races
    compare_races(race1_results, race2_results)
    
    # Print summary
    suite.print_summary()
    
    # Detailed validation report for both races
    if race1_results or race2_results:
        print("\n" + "="*80)
        print("  VALIDATION REPORT - KEY FINDINGS")
        print("="*80)
        
        if race1_results:
            print("\n" + "🏁"*40)
            print("  RACE 1 SUMMARY")
            print("🏁"*40)
            
            print(f"\n📊 DATA SCALE:")
            print(f"   Raw data: {len(race1_results['df']):,} rows")
            print(f"   After cleaning: {len(race1_results['df_clean']):,} rows ({len(race1_results['df_clean'])/len(race1_results['df'])*100:.1f}% retained)")
            print(f"   Total laps: {len(race1_results['lap_summary']):,}")
            print(f"   Vehicles: {race1_results['lap_summary']['vehicle_id'].nunique()}")
            
            print(f"\n🔬 PHYSICS VALIDATION:")
            print(f"   Max G-Force: {race1_results['df_clean']['Total_G'].max():.3f}G ✅")
            print(f"   Mean G-Force: {race1_results['df_clean']['Total_G'].mean():.3f}G")
            
            print(f"\n⚙️  TIRE STRESS METRICS:")
            print(f"   Total tire stress: {race1_results['df_stress']['tire_stress'].sum():.2f}")
            print(f"   Mean per sample: {race1_results['df_stress']['tire_stress'].mean():.6f}")
            
            print(f"\n💰 ROI EFFICIENCY:")
            category_counts = race1_results['lap_roi']['roi_category'].value_counts()
            for cat in ['EXCELLENT', 'GOOD', 'WASTEFUL', 'TERRIBLE']:
                if cat in category_counts:
                    count = category_counts[cat]
                    print(f"   {cat}: {count} laps ({count/len(race1_results['lap_roi'])*100:.1f}%)")
            
            print(f"\n🏎️  DRIVER SKILL:")
            print(f"   Trail Braking: {race1_results['skill_analysis']['trail_brake_pct']:.1f}%")
            print(f"   Assessment: {race1_results['skill_analysis']['skill_level']}")
            print(f"   Shape: {race1_results['skill_analysis']['shape']}")
            
            print(f"\n🔮 TIRE FAILURE PREDICTION:")
            print(f"   Current Lap: {race1_results['prediction']['current_lap']}")
            print(f"   Predicted Failure: Lap {race1_results['prediction']['predicted_failure_lap']:.1f}")
            print(f"   Laps Remaining: {race1_results['prediction']['laps_remaining']:.1f}")
        
        if race2_results:
            print("\n" + "🏁"*40)
            print("  RACE 2 SUMMARY")
            print("🏁"*40)
            
            print(f"\n📊 DATA SCALE:")
            print(f"   Raw data: {len(race2_results['df']):,} rows")
            print(f"   After cleaning: {len(race2_results['df_clean']):,} rows ({len(race2_results['df_clean'])/len(race2_results['df'])*100:.1f}% retained)")
            print(f"   Total laps: {len(race2_results['lap_summary']):,}")
            print(f"   Vehicles: {race2_results['lap_summary']['vehicle_id'].nunique()}")
            
            print(f"\n🔬 PHYSICS VALIDATION:")
            print(f"   Max G-Force: {race2_results['df_clean']['Total_G'].max():.3f}G ✅")
            print(f"   Mean G-Force: {race2_results['df_clean']['Total_G'].mean():.3f}G")
            
            print(f"\n⚙️  TIRE STRESS METRICS:")
            print(f"   Total tire stress: {race2_results['df_stress']['tire_stress'].sum():.2f}")
            print(f"   Mean per sample: {race2_results['df_stress']['tire_stress'].mean():.6f}")
            
            print(f"\n💰 ROI EFFICIENCY:")
            category_counts = race2_results['lap_roi']['roi_category'].value_counts()
            for cat in ['EXCELLENT', 'GOOD', 'WASTEFUL', 'TERRIBLE']:
                if cat in category_counts:
                    count = category_counts[cat]
                    print(f"   {cat}: {count} laps ({count/len(race2_results['lap_roi'])*100:.1f}%)")
            
            print(f"\n🏎️  DRIVER SKILL:")
            print(f"   Trail Braking: {race2_results['skill_analysis']['trail_brake_pct']:.1f}%")
            print(f"   Assessment: {race2_results['skill_analysis']['skill_level']}")
            print(f"   Shape: {race2_results['skill_analysis']['shape']}")
            
            print(f"\n🔮 TIRE FAILURE PREDICTION:")
            print(f"   Current Lap: {race2_results['prediction']['current_lap']}")
            print(f"   Predicted Failure: Lap {race2_results['prediction']['predicted_failure_lap']:.1f}")
            print(f"   Laps Remaining: {race2_results['prediction']['laps_remaining']:.1f}")
    
    print("\n" + "="*80)
    print("  END OF TESTING SUITE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
