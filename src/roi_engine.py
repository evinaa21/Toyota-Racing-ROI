"""
Toyota GR Cup - Racing ROI (Return on Investment) Engine
=========================================================
Calculates the "Cost per Corner" for every lap by comparing tire stress
(investment) against time gained (return).

Helps identify:
- OVERSPENDING: Destroying tires for minimal time gain
- UNDERSPENDING: Leaving performance on the table with fresh tires

Author: Toyota Racing ROI Analysis
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_and_pivot_telemetry(filepath):
    """
    Load telemetry data and pivot from long to wide format.
    
    Parameters:
    -----------
    filepath : str
        Path to telemetry CSV file
    
    Returns:
    --------
    pd.DataFrame
        Pivoted telemetry data
    """
    print("="*70)
    print("RACING ROI ENGINE - LOADING DATA")
    print("="*70)
    
    # DATA SOURCE VERIFICATION
    abs_path = os.path.abspath(filepath)
    print(f"📂 LOADING DATA FROM: {abs_path}")
    
    if "data" not in abs_path.lower():
        print("⚠️ WARNING: You are not loading from the standard 'data/' folder. Please check your path.")
    
    df_long = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df_long):,} rows")
    
    print("Pivoting to wide format...")
    df_wide = df_long.pivot_table(
        index=['timestamp', 'vehicle_id', 'vehicle_number', 'lap', 'outing'],
        columns='telemetry_name',
        values='telemetry_value',
        aggfunc='first'
    ).reset_index()
    
    df_wide.columns.name = None
    print(f"Pivoted to {len(df_wide):,} rows x {len(df_wide.columns)} columns")
    
    return df_wide

def clean_telemetry(df):
    """
    Clean telemetry data with Professional Signal Processing.
    
    IMPROVEMENT 1: SIGNAL PROCESSING
    Instead of hard dropping > 2.0G, we use Savitzky-Golay filters to 
    smooth sensor noise while preserving real transient peaks (curb strikes).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw telemetry data
    
    Returns:
    --------
    pd.DataFrame
        Cleaned data
    """
    print("\n" + "="*70)
    print("CLEANING TELEMETRY (PROFESSIONAL SIGNAL PROCESSING)")
    print("="*70)
    
    initial = len(df)
    
    # Convert to numeric
    for col in ['speed', 'accx_can', 'accy_can', 'Steering_Angle']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter: Speed >= 10 km/h (on track)
    df = df[df['speed'] >= 10].copy()
    print(f"After speed filter: {len(df):,} rows")
    
    # Filter: Valid acceleration data
    df = df[
        (df['accx_can'].notna()) & 
        (df['accy_can'].notna()) &
        (df['accx_can'] != 0) & 
        (df['accy_can'] != 0)
    ].copy()
    
    # --- SIGNAL PROCESSING UPGRADE ---
    print("Applying Savitzky-Golay Filter (Window=11, Poly=3)...")
    
    # We must apply filters per-lap to avoid smoothing across discontinuities
    # Using groupby().transform() for vectorized application
    def smooth_signal(x):
        if len(x) > 11:
            return savgol_filter(x, window_length=11, polyorder=3)
        return x

    # Group by vehicle and lap to respect physical boundaries
    # Note: This might take a moment but is essential for accuracy
    df['accx_can'] = df.groupby(['vehicle_id', 'lap'])['accx_can'].transform(smooth_signal)
    df['accy_can'] = df.groupby(['vehicle_id', 'lap'])['accy_can'].transform(smooth_signal)
    
    # We NO LONGER drop > 2.0G rows. 
    # The filter handles noise, and real high-G impacts (curbs) are preserved.
    
    print(f"Total removed: {initial - len(df):,} rows ({(initial-len(df))/initial*100:.1f}%)")
    
    return df

def calculate_tire_stress(df):
    """
    STEP 1: Calculate Tire Energy (Physics Upgrade) & Resample Data
    
    IMPROVEMENT 2: DATA ALIGNMENT
    Resamples all laps to a common 1m Distance Grid for accurate comparison.
    
    IMPROVEMENT 3: PHYSICS UPGRADE
    Calculates 'Tire Energy' (Work Done) instead of just Stress.
    Formula: Stress = (Total_G)^2 * Speed * Distance_Step
    
    IMPROVEMENT 4: CORNER DETECTION
    Auto-segments track into 'Cornering Zones'.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned telemetry
    
    Returns:
    --------
    pd.DataFrame
        Resampled data with tire_stress and corner_id
    """
    print("\n" + "="*70)
    print("STEP 1: PHYSICS ENGINE & DATA ALIGNMENT")
    print("="*70)
    
    # Pre-calculate speed in m/s
    df['speed_ms'] = df['speed'] / 3.6
    df['dt'] = 0.02  # 50Hz assumption
    df['ds'] = df['speed_ms'] * df['dt']
    
    resampled_laps = []
    
    print("Resampling laps to 1m Distance Grid & Detecting Corners...")
    
    # Process each lap individually
    for (vid, lap), group in df.groupby(['vehicle_id', 'lap']):
        if len(group) < 10: continue
            
        group = group.sort_values('timestamp')
        
        # Calculate cumulative distance for this lap
        group['dist_cum'] = group['ds'].cumsum()
        group['time_cum'] = group['dt'].cumsum()
        
        max_dist = group['dist_cum'].max()
        
        # Create common Distance Grid (1m increments)
        # This aligns all laps spatially!
        grid_dist = np.arange(0, max_dist, 1.0)
        
        if len(grid_dist) < 10: continue
            
        # Interpolate telemetry to the grid
        # We use 'extrapolate' to handle the very start/end
        f_speed = interp1d(group['dist_cum'], group['speed_ms'], kind='linear', fill_value="extrapolate")
        f_accx = interp1d(group['dist_cum'], group['accx_can'], kind='linear', fill_value="extrapolate")
        f_accy = interp1d(group['dist_cum'], group['accy_can'], kind='linear', fill_value="extrapolate")
        f_time = interp1d(group['dist_cum'], group['time_cum'], kind='linear', fill_value="extrapolate")
        
        new_speed = f_speed(grid_dist)
        new_accx = f_accx(grid_dist)
        new_accy = f_accy(grid_dist)
        new_time = f_time(grid_dist)
        
        # --- PHYSICS UPGRADE: TIRE ENERGY ---
        # Stress = (Total_G)^2 * Speed * Distance_Step
        # Distance_Step is 1.0m (from our grid)
        total_g = np.sqrt(new_accx**2 + new_accy**2)
        tire_energy = (total_g**2) * new_speed * 1.0
        
        # --- CORNER DETECTION ---
        # Corner = Lateral G > 0.5 for > 20m
        is_corner_candidate = np.abs(new_accy) > 0.5
        corner_ids = np.zeros(len(grid_dist), dtype=int)
        
        current_corner_id = 0
        in_corner = False
        start_idx = 0
        
        for i in range(len(grid_dist)):
            if is_corner_candidate[i]:
                if not in_corner:
                    in_corner = True
                    start_idx = i
            else:
                if in_corner:
                    in_corner = False
                    # Check length (1 index = 1 meter)
                    if (i - start_idx) > 20:
                        current_corner_id += 1
                        corner_ids[start_idx:i] = current_corner_id
        
        # Handle corner at end of lap
        if in_corner and (len(grid_dist) - start_idx) > 20:
            current_corner_id += 1
            corner_ids[start_idx:] = current_corner_id
            
        # Create Resampled DataFrame
        lap_df = pd.DataFrame({
            'vehicle_id': vid,
            'vehicle_number': group['vehicle_number'].iloc[0],
            'lap': lap,
            'dist': grid_dist,
            'time': new_time,
            'speed_ms': new_speed,
            'accx': new_accx,
            'accy': new_accy,
            'total_g': total_g,
            'tire_stress': tire_energy,
            'corner_id': corner_ids
        })
        
        resampled_laps.append(lap_df)
        
    df_resampled = pd.concat(resampled_laps, ignore_index=True)
    
    print(f"Processed {len(df_resampled):,} resampled data points")
    print(f"Physics Model: Tire Energy (Work Done)")
    print(f"Grid Resolution: 1.0 meter")
    
    return df_resampled

def calculate_lap_metrics(df):
    """
    STEP 2: Calculate per-lap metrics using High-Fidelity Data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Resampled telemetry (1m grid)
    
    Returns:
    --------
    pd.DataFrame
        Lap-level summary
    """
    print("\n" + "="*70)
    print("STEP 2: CALCULATING LAP METRICS (HIGH FIDELITY)")
    print("="*70)
    
    # Group by vehicle and lap
    lap_summary = df.groupby(['vehicle_id', 'vehicle_number', 'lap']).agg({
        'tire_stress': 'sum',            # Total Energy
        'total_g': 'mean',               # Avg G
        'speed_ms': 'mean',              # Avg Speed
        'time': 'max',                   # Lap Time (from interpolated time)
        'dist': 'max'                    # Lap Distance
    }).reset_index()
    
    # Rename for compatibility
    lap_summary.rename(columns={
        'lap_tire_stress': 'tire_stress', # It's already named tire_stress in agg
        'speed_ms': 'avg_speed',
        'time': 'lap_time_est'
    }, inplace=True)
    
    # Convert speed back to kph for display
    lap_summary['avg_speed'] = lap_summary['avg_speed'] * 3.6
    
    # Rename tire_stress to lap_tire_stress for compatibility
    lap_summary.rename(columns={'tire_stress': 'lap_tire_stress'}, inplace=True)
    
    # --- TIME DELTA CALCULATION ---
    # Compare against the session best lap
    # Since we are on a distance grid, we can compare times directly?
    # Actually, simple LapTime delta is sufficient for the summary.
    # But for "Time Delta" column, we want the gap to the best lap.
    
    # Find global best lap time (or per vehicle?)
    # Usually ROI is per-vehicle self-improvement.
    
    for vehicle in lap_summary['vehicle_id'].unique():
        mask = lap_summary['vehicle_id'] == vehicle
        vehicle_data = lap_summary[mask]
        
        # Reference Lap: Lowest Stress (Efficiency Benchmark)
        # FIX: Filter out slow "Out Laps" (> 130s) and incomplete laps (< 45s)
        # This ensures we only compare against valid racing laps
        valid_laps = vehicle_data[
            (vehicle_data['lap_time_est'] <= 130) & 
            (vehicle_data['lap_time_est'] > 45)
        ]
        
        if len(valid_laps) > 0:
            ref_lap_idx = valid_laps['lap_tire_stress'].idxmin()
        else:
            # Fallback if no valid laps found (use median duration lap?)
            # Or just min stress of whatever we have
            ref_lap_idx = vehicle_data['lap_tire_stress'].idxmin()
            
        ref_stress = lap_summary.loc[ref_lap_idx, 'lap_tire_stress']
        ref_time = lap_summary.loc[ref_lap_idx, 'lap_time_est']
        
        # SAFETY CHECK: Avoid division by zero
        if ref_stress == 0:
            ref_stress = 1.0
        
        # Calculate deltas
        lap_summary.loc[mask, 'stress_delta'] = lap_summary.loc[mask, 'lap_tire_stress'] - ref_stress
        lap_summary.loc[mask, 'time_delta'] = lap_summary.loc[mask, 'lap_time_est'] - ref_time
        
        # Percentages
        lap_summary.loc[mask, 'stress_delta_pct'] = (lap_summary.loc[mask, 'stress_delta'] / ref_stress) * 100
        # Handle division by zero for time if needed, but time is never 0
    
    print(f"Analyzed {len(lap_summary)} laps")
    print(f"Vehicles: {lap_summary['vehicle_number'].unique().tolist()}")
    
    return lap_summary

def calculate_roi_efficiency(lap_summary):
    """
    STEP 3: Calculate ROI (Return on Investment) Efficiency
    
    ROI = Time Gained / Tire Stress Invested
    
    Parameters:
    -----------
    lap_summary : pd.DataFrame
        Lap metrics
    
    Returns:
    --------
    pd.DataFrame
        Lap summary with ROI scores
    """
    print("\n" + "="*70)
    print("STEP 3: CALCULATING ROI EFFICIENCY")
    print("="*70)
    
    # RECALCULATE DELTAS WITH IMPROVED REFERENCE LAP LOGIC
    # We do this here to fix the "Out Lap" bug without rewriting the previous function
    
    for vehicle in lap_summary['vehicle_id'].unique():
        mask = lap_summary['vehicle_id'] == vehicle
        vehicle_data = lap_summary[mask].copy()
        
        # 1. FIX REFERENCE LAP SELECTION
        # Calculate median lap time to establish a baseline pace
        median_time = vehicle_data['lap_time_est'].median()
        
        # Filter for VALID RACING LAPS (The 107% Rule)
        # Exclude Out Laps, Yellow Flags, and Pit Laps
        valid_laps = vehicle_data[vehicle_data['lap_time_est'] <= (median_time * 1.07)]
        
        if len(valid_laps) > 0:
            # CHANGE: Use MEDIAN lap as baseline so we have ~50% Green bars
            # Comparing to the fastest lap made everyone look "Terrible"
            median_time = valid_laps['lap_time_est'].median()
            # Find lap closest to median time
            ref_lap_idx = (valid_laps['lap_time_est'] - median_time).abs().idxmin()
            ref_type = "MEDIAN (BASELINE)"
        else:
            # Fallback: Use the median lap if no clean laps exist
            ref_lap_idx = (vehicle_data['lap_time_est'] - median_time).abs().idxmin()
            ref_type = "MEDIAN (FALLBACK)"
            
        ref_lap = lap_summary.loc[ref_lap_idx]
        ref_stress = ref_lap['lap_tire_stress']
        ref_time = ref_lap['lap_time_est']
        
        # 4. SAFETY CHECK: Print reference details
        print(f"🏎️ Reference Lap Selected: Lap {int(ref_lap['lap'])} ({ref_type}) | Time: {ref_time:.2f}s | Stress: {ref_stress:,.0f}")
        
        # 3. HANDLE DIVISION BY ZERO
        if ref_stress == 0:
            ref_stress = 1.0
            
        # Recalculate deltas
        lap_summary.loc[mask, 'stress_delta'] = lap_summary.loc[mask, 'lap_tire_stress'] - ref_stress
        lap_summary.loc[mask, 'time_delta'] = lap_summary.loc[mask, 'lap_time_est'] - ref_time
        lap_summary.loc[mask, 'stress_delta_pct'] = (lap_summary.loc[mask, 'stress_delta'] / ref_stress) * 100

    # ROI Efficiency Score
    # Negative time delta = faster (good)
    # Lower stress delta = less wear (good)
    
    # Avoid division by zero
    lap_summary['stress_delta_safe'] = lap_summary['stress_delta'].replace(0, 1.0)
    
    # 2. SCALE THE ROI SCORE
    # Scale by 5000 for better visibility and "bigger bars"
    lap_summary['roi_efficiency'] = (-lap_summary['time_delta'] / lap_summary['stress_delta_safe'].abs()) * 5000
    
    # Categorize efficiency
    def categorize_roi(roi):
        if pd.isna(roi) or np.isinf(roi):
            return 'REFERENCE'
        elif roi > 1.0:
            return 'EXCELLENT'
        elif roi > 0:
            return 'GOOD'
        elif roi > -1.0:
            return 'WASTEFUL'
        else:
            return 'TERRIBLE'
    
    lap_summary['roi_category'] = lap_summary['roi_efficiency'].apply(categorize_roi)
    
    print(f"\nROI Distribution:")
    print(lap_summary['roi_category'].value_counts())
    
    return lap_summary

def predict_tire_failure(lap_summary, vehicle_id):
    """
    GENIUS TWIST: Predict when tires will fail based on stress accumulation
    
    Uses linear regression on cumulative tire stress
    
    Parameters:
    -----------
    lap_summary : pd.DataFrame
        Lap metrics
    vehicle_id : str
        Vehicle to analyze
    
    Returns:
    --------
    dict
        Prediction results
    """
    print("\n" + "="*70)
    print("GENIUS TWIST: TIRE FAILURE PREDICTION")
    print("="*70)
    
    # Filter to specific vehicle
    vehicle_data = lap_summary[lap_summary['vehicle_id'] == vehicle_id].copy()
    vehicle_data = vehicle_data.sort_values('lap')
    
    # Calculate cumulative tire stress
    vehicle_data['cumulative_stress'] = vehicle_data['lap_tire_stress'].cumsum()
    
    # Prepare data for regression
    X = vehicle_data['lap'].values.reshape(-1, 1)
    y = vehicle_data['cumulative_stress'].values
    
    # Train linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Define failure threshold (Dynamic based on driving intensity)
    # We assume tires can handle ~25 laps of "average" driving before degradation becomes critical
    # This scales automatically whether we use Stress (G^2*t) or Energy (G^2*v*d)
    avg_lap_stress = vehicle_data['lap_tire_stress'].mean()
    FAILURE_THRESHOLD = avg_lap_stress * 25
    
    # Predict failure lap
    failure_lap = (FAILURE_THRESHOLD - model.intercept_) / model.coef_[0]
    
    # Current status
    current_lap = vehicle_data['lap'].max()
    current_stress = vehicle_data['cumulative_stress'].max()
    stress_rate = model.coef_[0]
    
    print(f"Vehicle: {vehicle_id}")
    print(f"Current Lap: {current_lap}")
    print(f"Cumulative Stress: {current_stress:.2f}")
    print(f"Stress Rate: {stress_rate:.2f} per lap")
    print(f"Predicted Failure Lap: {failure_lap:.1f}")
    print(f"Laps Remaining: {failure_lap - current_lap:.1f}")
    
    return {
        'vehicle_id': vehicle_id,
        'current_lap': current_lap,
        'cumulative_stress': current_stress,
        'stress_rate': stress_rate,
        'failure_threshold': FAILURE_THRESHOLD,
        'predicted_failure_lap': failure_lap,
        'laps_remaining': failure_lap - current_lap,
        'model': model,
        'data': vehicle_data
    }

def generate_coaching_advice(lap_summary, vehicle_id):
    """
    STEP 4: Generate coaching recommendations
    
    Parameters:
    -----------
    lap_summary : pd.DataFrame
        Lap metrics with ROI
    vehicle_id : str
        Vehicle to analyze
    
    Returns:
    --------
    list
        Coaching recommendations
    """
    print("\n" + "="*70)
    print("STEP 4: GENERATING COACHING ADVICE")
    print("="*70)
    
    vehicle_data = lap_summary[lap_summary['vehicle_id'] == vehicle_id].copy()
    advice = []
    
    # Find wasteful laps (high stress, slow time)
    wasteful = vehicle_data[vehicle_data['roi_category'].isin(['WASTEFUL', 'TERRIBLE'])]
    
    if len(wasteful) > 0:
        for _, lap in wasteful.iterrows():
            msg = f"⚠️ LAP {int(lap['lap'])} WARNING:\n"
            msg += f"   Tire Stress: +{lap['stress_delta_pct']:.1f}% vs reference\n"
            msg += f"   Time Gained: {lap['time_delta']:.2f}s (slower!)\n"
            msg += f"   ROI Score: {lap['roi_efficiency']:.3f} (WASTEFUL)\n"
            msg += f"   💡 RECOMMENDATION: You're destroying tires for NO speed gain.\n"
            msg += f"      Focus on smooth inputs and trail braking technique.\n"
            advice.append(msg)
    
    # Find efficient laps
    efficient = vehicle_data[vehicle_data['roi_category'] == 'EXCELLENT']
    if len(efficient) > 0:
        best_lap = efficient.loc[efficient['roi_efficiency'].idxmax()]
        msg = f"✅ LAP {int(best_lap['lap'])} - EXCELLENT EFFICIENCY:\n"
        msg += f"   This is your benchmark! Study this lap's data.\n"
        msg += f"   ROI Score: {best_lap['roi_efficiency']:.3f}\n"
        advice.append(msg)
    
    # Print advice
    for i, adv in enumerate(advice, 1):
        print(f"\n{i}. {adv}")
    
    return advice

def create_roi_dashboard(lap_summary, prediction, save_path='roi_dashboard.png'):
    """
    Create comprehensive ROI dashboard visualization
    
    Parameters:
    -----------
    lap_summary : pd.DataFrame
        Lap metrics
    prediction : dict
        Tire failure prediction
    save_path : str
        Output path
    """
    print("\n" + "="*70)
    print("CREATING ROI DASHBOARD")
    print("="*70)
    
    vehicle_data = prediction['data']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Tire Stress per Lap
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(vehicle_data['lap'], vehicle_data['lap_tire_stress'], 
            color='orangered', alpha=0.7, edgecolor='black')
    ax1.axhline(vehicle_data['lap_tire_stress'].mean(), 
                color='blue', linestyle='--', label='Average Stress')
    ax1.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tire Stress (Investment)', fontsize=12, fontweight='bold')
    ax1.set_title('Tire Stress per Lap - The "Cost"', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Lap Time Delta
    ax2 = fig.add_subplot(gs[1, :2])
    colors = ['green' if x < 0 else 'red' for x in vehicle_data['time_delta']]
    ax2.bar(vehicle_data['lap'], vehicle_data['time_delta'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time Delta vs Reference (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Time Gained/Lost - The "Return"', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. ROI Efficiency Score
    ax3 = fig.add_subplot(gs[2, :2])
    roi_colors = vehicle_data['roi_efficiency'].apply(
        lambda x: 'green' if x > 0.5 else 'yellow' if x > 0 else 'red'
    )
    ax3.bar(vehicle_data['lap'], vehicle_data['roi_efficiency'], 
            color=roi_colors, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=2)
    ax3.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('ROI Efficiency', fontsize=12, fontweight='bold')
    ax3.set_title('ROI Score: Time Gained / Tire Stress', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Tire Failure Prediction
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.plot(vehicle_data['lap'], vehicle_data['cumulative_stress'], 
             'o-', color='darkblue', linewidth=2, markersize=8, label='Actual')
    
    # Prediction line
    future_laps = np.arange(1, prediction['predicted_failure_lap'] + 5).reshape(-1, 1)
    future_stress = prediction['model'].predict(future_laps)
    ax4.plot(future_laps, future_stress, '--', color='red', linewidth=2, label='Predicted')
    
    # Failure threshold
    ax4.axhline(prediction['failure_threshold'], 
                color='red', linestyle=':', linewidth=3, label='Failure Threshold')
    ax4.axvline(prediction['predicted_failure_lap'], 
                color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    # Annotation
    ax4.text(prediction['predicted_failure_lap'], prediction['failure_threshold']*0.9, 
             f'Predicted Failure:\nLap {prediction["predicted_failure_lap"]:.1f}',
             fontsize=11, fontweight='bold', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax4.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Tire Stress', fontsize=12, fontweight='bold')
    ax4.set_title('🔮 TIRE FAILURE PREDICTION\n(Genius Twist)', 
                  fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Racing ROI Engine - Vehicle {prediction["vehicle_id"]}\n' + 
                 f'Efficiency Analysis & Tire Life Prediction',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved: {save_path}")
    plt.show()

def main():
    """
    Main execution - Racing ROI Engine
    """
    print("\n" + "="*70)
    print("🏎️  TOYOTA GR CUP - RACING ROI ENGINE  🏎️")
    print("="*70)
    print("Analyzing tire management efficiency and predicting pit strategy")
    print("="*70 + "\n")
    
    # Load data
    filepath = r'data\Sonoma\Race 1\sonoma.csv'
    df = load_and_pivot_telemetry(filepath)
    
    # Clean data
    df = clean_telemetry(df)
    
    # Calculate tire stress
    df = calculate_tire_stress(df)
    
    # Calculate lap metrics
    lap_summary = calculate_lap_metrics(df)
    
    # Calculate ROI efficiency
    lap_summary = calculate_roi_efficiency(lap_summary)
    
    # Save lap summary
    lap_summary.to_csv('lap_roi_analysis.csv', index=False)
    print(f"\n✅ Lap analysis saved: lap_roi_analysis.csv")
    
    # Select vehicle for detailed analysis (first vehicle)
    vehicle_id = lap_summary['vehicle_id'].iloc[0]
    
    # Generate coaching advice
    advice = generate_coaching_advice(lap_summary, vehicle_id)
    
    # Predict tire failure
    prediction = predict_tire_failure(lap_summary, vehicle_id)
    
    # Create dashboard
    create_roi_dashboard(lap_summary, prediction, save_path='roi_dashboard.png')
    
    print("\n" + "="*70)
    print("🏆 ROI ENGINE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"📊 Dashboard: roi_dashboard.png")
    print(f"📈 Data: lap_roi_analysis.csv")
    print(f"🎯 Coaching insights: {len(advice)} recommendations generated")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
