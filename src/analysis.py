"""
Toyota GR Cup Telemetry Analysis Tool
======================================
This script analyzes racing telemetry data and creates a Tire Friction Circle
visualization to help understand the driver's grip limits.

Author: Toyota Racing ROI Analysis
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 10)

def load_telemetry_data(filepath):
    """
    Load telemetry data from CSV file and pivot from long to wide format.
    
    The data comes in long format with 'telemetry_name' and 'telemetry_value' columns.
    We need to pivot it to wide format for easier analysis.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing telemetry data
    
    Returns:
    --------
    pd.DataFrame
        Loaded and pivoted telemetry data
    """
    print(f"Loading data from: {filepath}")
    df_long = pd.read_csv(filepath, low_memory=False)
    print(f"Data loaded successfully! Shape (long format): {df_long.shape}")
    
    print("\nPivoting data from long to wide format...")
    print("This may take a moment for large datasets...")
    
    # Pivot the data: each telemetry_name becomes a column
    df_wide = df_long.pivot_table(
        index=['timestamp', 'vehicle_id', 'vehicle_number', 'lap', 'outing'],
        columns='telemetry_name',
        values='telemetry_value',
        aggfunc='first'  # Take first value if duplicates exist
    ).reset_index()
    
    # Flatten column names
    df_wide.columns.name = None
    
    print(f"Data pivoted successfully! Shape (wide format): {df_wide.shape}")
    
    return df_wide

def inspect_columns(df):
    """
    Print column names and basic info about the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry dataframe
    """
    print("\n" + "="*60)
    print("DATASET COLUMNS")
    print("="*60)
    print(df.columns.tolist())
    print("\n" + "="*60)
    print("DATASET INFO")
    print("="*60)
    print(df.info())
    print("\n" + "="*60)
    print("FIRST FEW ROWS")
    print("="*60)
    print(df.head())

def clean_telemetry_data(df):
    """
    Clean and filter telemetry data.
    
    Removes:
    - Rows where speed < 10 km/h (pit lane/stopped)
    - Rows where accx_can or accy_can are 0 or empty
    - SUPERHUMAN G-FORCES: Anything above ±2.0 G (sensor errors/noise)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw telemetry dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned telemetry dataframe
    """
    print("\n" + "="*60)
    print("CLEANING DATA")
    print("="*60)
    
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")
    
    # Convert relevant columns to numeric, handling any non-numeric values
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['accx_can'] = pd.to_numeric(df['accx_can'], errors='coerce')
    df['accy_can'] = pd.to_numeric(df['accy_can'], errors='coerce')
    
    # Filter out low speed (pit lane/stopped)
    df_clean = df[df['speed'] >= 10].copy()
    print(f"After removing speed < 10 km/h: {len(df_clean)} rows")
    
    # Filter out zero or empty acceleration values
    df_clean = df_clean[
        (df_clean['accx_can'].notna()) & 
        (df_clean['accy_can'].notna()) &
        (df_clean['accx_can'] != 0) & 
        (df_clean['accy_can'] != 0)
    ].copy()
    print(f"After removing invalid acceleration values: {len(df_clean)} rows")
    
    # STEP 1: Filter out superhuman G-forces (noise/sensor errors)
    # Real GR86 Cup cars can't pull more than 2.0 G in any direction
    df_clean = df_clean[
        (df_clean['accx_can'].abs() <= 2.0) & 
        (df_clean['accy_can'].abs() <= 2.0)
    ].copy()
    print(f"After removing superhuman G-forces (>2.0G): {len(df_clean)} rows")
    
    removed_rows = initial_rows - len(df_clean)
    print(f"Total rows removed: {removed_rows} ({removed_rows/initial_rows*100:.2f}%)")
    
    return df_clean

def analyze_driver_skill(df):
    """
    Analyze driver skill level based on friction circle shape.
    
    STEP 2: Interpret the Shape
    - Diamond Shape = Amateur driver (brake, then turn, then accelerate)
    - Mushroom/Circle Shape = Pro driver (trail braking - mixing inputs)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned telemetry dataframe with acceleration data
    
    Returns:
    --------
    dict
        Analysis metrics
    """
    print("\n" + "="*60)
    print("DRIVER SKILL ANALYSIS")
    print("="*60)
    
    # Calculate corner usage (points in each quadrant)
    combined_braking_left = len(df[(df['accx_can'] > 0.3) & (df['accy_can'] < -0.3)])
    combined_braking_right = len(df[(df['accx_can'] > 0.3) & (df['accy_can'] > 0.3)])
    combined_accel_left = len(df[(df['accx_can'] < -0.3) & (df['accy_can'] < -0.3)])
    combined_accel_right = len(df[(df['accx_can'] < -0.3) & (df['accy_can'] > 0.3)])
    
    # Straight line actions
    pure_braking = len(df[(df['accx_can'].abs() > 0.5) & (df['accy_can'].abs() < 0.2)])
    pure_turning = len(df[(df['accy_can'].abs() > 0.5) & (df['accx_can'].abs() < 0.2)])
    
    total_points = len(df)
    combined_total = combined_braking_left + combined_braking_right + combined_accel_left + combined_accel_right
    
    # Calculate "trail braking" percentage
    trail_brake_pct = (combined_total / total_points * 100) if total_points > 0 else 0
    
    print(f"Trail Braking Usage (combined inputs): {trail_brake_pct:.1f}%")
    print(f"  - Braking + Left Turn: {combined_braking_left} points")
    print(f"  - Braking + Right Turn: {combined_braking_right} points")
    print(f"  - Accel + Left Turn: {combined_accel_left} points")
    print(f"  - Accel + Right Turn: {combined_accel_right} points")
    print(f"\nSeparate Actions:")
    print(f"  - Pure Braking: {pure_braking} points")
    print(f"  - Pure Turning: {pure_turning} points")
    
    # Skill assessment
    if trail_brake_pct > 40:
        skill_level = "PRO - Excellent trail braking technique!"
        shape = "MUSHROOM/CIRCLE"
    elif trail_brake_pct > 25:
        skill_level = "INTERMEDIATE - Good corner entry"
        shape = "ROUNDED DIAMOND"
    else:
        skill_level = "AMATEUR - Sequential inputs (brake, turn, gas)"
        shape = "DIAMOND"
    
    print(f"\n{'='*60}")
    print(f"ASSESSMENT: {skill_level}")
    print(f"Expected Shape: {shape}")
    print(f"{'='*60}")
    
    return {
        'trail_brake_pct': trail_brake_pct,
        'skill_level': skill_level,
        'shape': shape
    }

def calculate_total_g(df):
    """
    Calculate total G-force from longitudinal and lateral acceleration.
    
    Formula: Total_G = sqrt(accx_can^2 + accy_can^2)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned telemetry dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added Total_G column
    """
    print("\n" + "="*60)
    print("CALCULATING TOTAL G-FORCE")
    print("="*60)
    
    df['Total_G'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)
    
    print(f"Total_G statistics:")
    print(f"  Mean: {df['Total_G'].mean():.3f} G")
    print(f"  Max:  {df['Total_G'].max():.3f} G")
    print(f"  Min:  {df['Total_G'].min():.3f} G")
    
    return df
    """
    Calculate total G-force from longitudinal and lateral acceleration.
    
    Formula: Total_G = sqrt(accx_can^2 + accy_can^2)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned telemetry dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added Total_G column
    """
    print("\n" + "="*60)
    print("CALCULATING TOTAL G-FORCE")
    print("="*60)
    
    df['Total_G'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)
    
    print(f"Total_G statistics:")
    print(f"  Mean: {df['Total_G'].mean():.3f} G")
    print(f"  Max:  {df['Total_G'].max():.3f} G")
    print(f"  Min:  {df['Total_G'].min():.3f} G")
    
    return df

def create_friction_circle(df, skill_analysis, save_path='friction_circle.png'):
    """
    Create a Tire Friction Circle visualization.
    
    The friction circle shows the relationship between lateral and longitudinal
    acceleration, helping visualize the grip limits of the vehicle.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Telemetry dataframe with acceleration data
    skill_analysis : dict
        Driver skill analysis results
    save_path : str
        Path to save the output plot
    """
    print("\n" + "="*60)
    print("CREATING FRICTION CIRCLE VISUALIZATION")
    print("="*60)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create scatter plot with speed as color
    scatter = ax.scatter(
        df['accy_can'],           # X-axis: Lateral G
        df['accx_can'],           # Y-axis: Longitudinal G
        c=df['speed'],            # Color by speed
        cmap='inferno',           # Colormap
        alpha=0.6,                # Transparency
        s=2,                      # Point size (slightly larger)
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (km/h)', rotation=270, labelpad=20, fontsize=12)
    
    # Draw reference circles
    circle_15 = plt.Circle((0, 0), 1.5, color='lime', fill=False, 
                       linewidth=2, linestyle='--', 
                       label='1.5G (Pro Limit)', alpha=0.7)
    ax.add_patch(circle_15)
    
    circle_10 = plt.Circle((0, 0), 1.0, color='cyan', fill=False, 
                       linewidth=1, linestyle=':', 
                       label='1.0G (Comfort)', alpha=0.5)
    ax.add_patch(circle_10)
    
    # Set axis limits to zoom into realistic range
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    
    # Add grid lines at major intervals
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.axhline(y=0, color='white', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color='white', linewidth=1.5, alpha=0.7)
    
    # Labels and title with skill assessment
    ax.set_xlabel('Lateral G-Force (accy_can)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Longitudinal G-Force (accx_can)', fontsize=14, fontweight='bold')
    
    title = f'Toyota GR Cup - Tire Friction Circle\n'
    title += f'Skill Level: {skill_analysis["skill_level"]}\n'
    title += f'Expected Shape: {skill_analysis["shape"]} | Trail Braking: {skill_analysis["trail_brake_pct"]:.1f}%'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Ensure equal aspect ratio for circular appearance
    ax.set_aspect('equal', adjustable='box')
    
    # Add annotations for quadrants with driving technique
    ax.text(1.5, 1.5, 'Right Turn\n+ Braking\n(Trail Brake)', 
           fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.text(-1.5, 1.5, 'Left Turn\n+ Braking\n(Trail Brake)', 
           fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.text(1.5, -1.5, 'Right Turn\n+ Accel\n(Power On)', 
           fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    ax.text(-1.5, -1.5, 'Left Turn\n+ Accel\n(Power On)', 
           fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    # Add interpretation guide
    guide_text = "Shape Guide:\n"
    guide_text += "• Diamond = Amateur (brake→turn→gas)\n"
    guide_text += "• Mushroom = Pro (mixed inputs)"
    ax.text(0.02, 0.98, guide_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Display plot
    plt.show()

def main():
    """
    Main execution function.
    """
    print("="*60)
    print("TOYOTA GR CUP TELEMETRY ANALYSIS")
    print("="*60)
    
    # Define file path - UPDATE THIS PATH TO YOUR CSV FILE
    filepath = r'data\Sonoma\Race 1\sonoma.csv'
    
    try:
        # Step 1: Load data
        df = load_telemetry_data(filepath)
        
        # Step 2: Inspect columns
        inspect_columns(df)
        
        # Step 3: Clean data (includes 2.0G filter for realistic physics)
        df_clean = clean_telemetry_data(df)
        
        # Step 4: Calculate Total G
        df_clean = calculate_total_g(df_clean)
        
        # Step 5: Analyze driver skill (STEP 2: Diamond vs Mushroom)
        skill_analysis = analyze_driver_skill(df_clean)
        
        # Step 6: Create friction circle visualization
        create_friction_circle(df_clean, skill_analysis, save_path='friction_circle.png')
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nERROR: File not found at {filepath}")
        print("Please update the 'filepath' variable in the main() function")
        print("with the correct path to your CSV file.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
