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
from scipy import interpolate
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
    print(f"Loading: {filepath}")
    
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
    Clean telemetry data with realistic physics filters.
    
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
    print("CLEANING TELEMETRY")
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
    print(f"After acceleration filter: {len(df):,} rows")
    
    # Filter: Realistic G-forces (≤2.0G)
    df = df[
        (df['accx_can'].abs() <= 2.0) & 
        (df['accy_can'].abs() <= 2.0)
    ].copy()
    print(f"After G-force filter: {len(df):,} rows")
    
    print(f"Total removed: {initial - len(df):,} rows ({(initial-len(df))/initial*100:.1f}%)")
    
    return df

def calculate_tire_stress(df):
    """
    STEP 1: Calculate Tire Stress Metric
    
    Formula: Stress = (Lateral_G² + Longitudinal_G²) * Duration
    High G-force sustained over time = High tire degradation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned telemetry
    
    Returns:
    --------
    pd.DataFrame
        Data with tire_stress column
    """
    print("\n" + "="*70)
    print("STEP 1: CALCULATING TIRE STRESS")
    print("="*70)
    
    # Calculate Total G-Force
    df['total_g'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)
    
    # Calculate time delta between samples (assume ~50Hz sampling)
    df['time_delta'] = 0.02  # 20ms per sample (50Hz)
    
    # Tire Stress = G² * Duration
    # Squaring emphasizes high-G maneuvers (exponential tire wear)
    df['tire_stress'] = (df['total_g']**2) * df['time_delta']
    
    print(f"Tire Stress Statistics:")
    print(f"  Mean:   {df['tire_stress'].mean():.4f}")
    print(f"  Median: {df['tire_stress'].median():.4f}")
    print(f"  Max:    {df['tire_stress'].max():.4f}")
    print(f"  Total Stress (all laps): {df['tire_stress'].sum():.2f}")
    
    return df

def calculate_lap_metrics(df):
    """
    STEP 2: Calculate per-lap metrics for ROI analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry with tire_stress
    
    Returns:
    --------
    pd.DataFrame
        Lap-level summary
    """
    print("\n" + "="*70)
    print("STEP 2: CALCULATING LAP METRICS")
    print("="*70)
    
    # Group by vehicle and lap
    lap_summary = df.groupby(['vehicle_id', 'vehicle_number', 'lap']).agg({
        'tire_stress': 'sum',           # Total stress per lap
        'total_g': 'mean',               # Average G-force
        'speed': 'mean',                 # Average speed
        'timestamp': 'count'             # Sample count (proxy for lap time)
    }).reset_index()
    
    lap_summary.columns = ['vehicle_id', 'vehicle_number', 'lap', 
                           'lap_tire_stress', 'avg_g', 'avg_speed', 'sample_count']
    
    # Estimate lap time (samples * 0.02 seconds)
    lap_summary['lap_time_est'] = lap_summary['sample_count'] * 0.02
    
    # Calculate delta to reference lap (lap with minimum stress)
    for vehicle in lap_summary['vehicle_id'].unique():
        mask = lap_summary['vehicle_id'] == vehicle
        vehicle_data = lap_summary[mask]
        
        # Find reference lap (lowest stress, likely smoothest driving)
        ref_lap_idx = vehicle_data['lap_tire_stress'].idxmin()
        ref_stress = lap_summary.loc[ref_lap_idx, 'lap_tire_stress']
        ref_time = lap_summary.loc[ref_lap_idx, 'lap_time_est']
        
        # Calculate deltas
        lap_summary.loc[mask, 'stress_delta'] = lap_summary.loc[mask, 'lap_tire_stress'] - ref_stress
        lap_summary.loc[mask, 'time_delta'] = lap_summary.loc[mask, 'lap_time_est'] - ref_time
        lap_summary.loc[mask, 'stress_delta_pct'] = (lap_summary.loc[mask, 'stress_delta'] / ref_stress) * 100
        lap_summary.loc[mask, 'time_delta_pct'] = (lap_summary.loc[mask, 'time_delta'] / ref_time) * 100
    
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
    
    # ROI Efficiency Score
    # Negative time delta = faster (good)
    # Lower stress delta = less wear (good)
    # We want: Maximum speed gain for minimum stress cost
    
    # Avoid division by zero
    lap_summary['stress_delta_safe'] = lap_summary['stress_delta'].replace(0, 0.001)
    
    # ROI = -time_delta / stress_delta
    # Negative because faster lap = negative time delta
    # Higher ROI = More efficient (faster with less tire wear)
    lap_summary['roi_efficiency'] = -lap_summary['time_delta'] / lap_summary['stress_delta_safe'].abs()
    
    # Categorize efficiency
    def categorize_roi(roi):
        if pd.isna(roi) or np.isinf(roi):
            return 'REFERENCE'
        elif roi > 0.5:
            return 'EXCELLENT'
        elif roi > 0:
            return 'GOOD'
        elif roi > -0.5:
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
    
    # Define failure threshold (arbitrary: 200 cumulative stress units)
    FAILURE_THRESHOLD = 200
    
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
    filepath = r'data\Sonoma\Race 1\sonoma_telemetry_R1.csv'
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
