# 🏁 Toyota GR Cup - Racing ROI Dashboard

> **Professional Software Development Project**  
> Advanced telemetry analysis and tire management optimization for Toyota GR Cup Racing

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Analysis Modules](#analysis-modules)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Toyota GR Cup Racing ROI Dashboard** is a comprehensive data analysis platform that transforms raw racing telemetry into actionable insights for tire management, driver skill assessment, and strategic pit stop planning. Built for the Toyota/TRD Hackathon competition, this project combines physics-based analysis with machine learning to optimize racing performance.

### Key Value Propositions

✅ **Speaks Toyota's Language** - Focus on tire management and efficiency  
✅ **Quantifiable Results** - Math-based analysis, not opinions  
✅ **Actionable Intelligence** - Specific lap-by-lap recommendations  
✅ **Strategic Value** - ML-powered pit stop predictions  
✅ **Professional Presentation** - Publication-ready visualizations

---

## 🚀 Features

### 1. **Friction Circle Analysis** 🎯
- Visualizes vehicle grip limits through G-force plotting
- Analyzes driver skill based on input technique
- Identifies Diamond (amateur) vs Mushroom (pro) driving patterns
- Calculates trail braking percentage (13.1% = Amateur, 40%+ = Pro)

### 2. **ROI Efficiency Scoring** 💰
- Calculates Return on Investment: `ROI = Time Gained / Tire Stress`
- Categories: EXCELLENT, GOOD, WASTEFUL, TERRIBLE
- Identifies laps where tires are destroyed for no speed gain
- Provides coaching recommendations for wasteful behavior

### 3. **Tire Failure Prediction** 🔮
- Uses Linear Regression on cumulative tire stress
- Predicts exact lap number for tire failure
- Enables proactive pit stop strategy
- Real-time stress rate monitoring (8.85 units/lap)

### 4. **Interactive Dashboard** 📊
- Built with Streamlit for real-time analysis
- Multi-race comparison support (Race 1 vs Race 2)
- Vehicle-specific deep dives
- Downloadable CSV reports

### 5. **Comprehensive Testing** ✅
- 94.9% test pass rate across 156 tests
- Validates all physics calculations
- Tests both Race 1 and Race 2 data
- Edge case handling verified

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                    │
│            (Interactive Web Interface)                  │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼─────────┐
│  Analysis.py   │    │  ROI_Engine.py   │
│ (Friction      │    │ (Tire Stress &   │
│  Circle)       │    │  ROI Analysis)   │
└───────┬────────┘    └────────┬─────────┘
        │                      │
        └──────────┬───────────┘
                   │
        ┌──────────▼──────────┐
        │   Data Pipeline     │
        │ • Load & Pivot      │
        │ • Clean & Filter    │
        │ • Calculate Metrics │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Raw Telemetry CSV  │
        │ • 27M+ rows         │
        │ • Long format       │
        │ • Multi-vehicle     │
        └─────────────────────┘
```

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher
- Git
- 8GB+ RAM recommended (for large datasets)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/evinaa21/Toyota-Racing-ROI.git
cd Toyota-Racing-ROI

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the Streamlit dashboard
streamlit run src/streamlit_app.py
```

### Alternative: Run Analysis Scripts

```bash
# Friction Circle Analysis
python src/analysis.py

# ROI Engine Analysis
python src/roi_engine.py

# Run Comprehensive Tests
python src/test_analysis.py
```

---

## 📁 Project Structure

```
Toyota-Racing-ROI/
│
├── data/                          # Telemetry data (gitignored)
│   └── Sonoma/
│       ├── Race 1/
│       │   └── sonoma_telemetry_R1.csv
│       └── Race 2/
│           └── sonoma_telemetry_R2.csv
│
├── src/                           # Source code
│   ├── streamlit_app.py          # 🎯 Main dashboard application
│   ├── analysis.py               # Friction circle analysis
│   ├── roi_engine.py             # ROI calculation engine
│   └── test_analysis.py          # Comprehensive test suite
│
├── notebooks/                     # Jupyter exploration (optional)
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git exclusions
```

---

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11**: Primary programming language
- **Streamlit 1.28+**: Interactive web dashboard framework
- **Pandas 2.3+**: Data manipulation and analysis
- **NumPy 2.3+**: Numerical computing and physics calculations
- **Matplotlib 3.10+**: Data visualization
- **Seaborn 0.13+**: Statistical data visualization

### Machine Learning
- **Scikit-learn 1.7+**: Linear regression for tire failure prediction
- **SciPy 1.16+**: Scientific computing and interpolation

### Development Tools
- **Git**: Version control
- **Virtual Environment**: Isolated Python environment
- **pytest**: Testing framework (optional)

---

## 🔬 Analysis Modules

### Module 1: `analysis.py` - Friction Circle Analysis

**Purpose**: Analyze driver skill through G-force visualization

**Key Functions**:
```python
load_telemetry_data(filepath)      # Pivot long→wide format
clean_telemetry_data(df)           # Filter noise & outliers
calculate_total_g(df)              # Total_G = √(accx² + accy²)
analyze_driver_skill(df)           # Trail braking % analysis
create_friction_circle(df, skill)  # Visualization
```

**Physics Formulas**:
- Total G-Force: `G = √(Longitudinal² + Lateral²)`
- G-Force Limits: ≤2.0G (realistic GR86 limits)
- Trail Braking: % of time with combined brake+turn inputs

**Output**: `friction_circle.png` (300 DPI)

---

### Module 2: `roi_engine.py` - Racing ROI Engine

**Purpose**: Calculate tire management efficiency and predict pit strategy

**Key Functions**:
```python
calculate_tire_stress(df)          # Stress = G² × Duration
calculate_lap_metrics(df)          # Per-lap aggregation
calculate_roi_efficiency(lap_sum)  # ROI = -time_delta / stress_delta
predict_tire_failure(lap_sum, vid) # ML prediction (LinearRegression)
generate_coaching_advice(lap_sum)  # AI recommendations
create_roi_dashboard(lap_sum)      # 4-panel visualization
```

**Physics Formulas**:
- Tire Stress: `Stress = G² × Time` (exponential wear)
- ROI: `ROI = Time Gained / Tire Stress Invested`
- Failure Prediction: Linear regression on cumulative stress

**Output**: 
- `roi_dashboard.png` (4-panel visualization)
- `lap_roi_analysis.csv` (682 lap records)

---

### Module 3: `streamlit_app.py` - Interactive Dashboard

**Purpose**: Web-based interface for real-time analysis

**Features**:
- 📊 **Overview Tab**: Race statistics and KPIs
- 🎯 **Friction Circle Tab**: Interactive G-force visualization
- 💰 **ROI Analysis Tab**: Efficiency distribution and top/worst laps
- 🔮 **Tire Failure Tab**: ML predictions with pit recommendations
- 🏎️ **Vehicle Details Tab**: Per-vehicle deep dive with coaching

**Caching**: Uses `@st.cache_data` for performance optimization

---

### Module 4: `test_analysis.py` - Comprehensive Testing

**Purpose**: Validate all calculations and formulas

**Test Coverage**:
- ✅ Data loading & pivoting (27M→3.6M rows)
- ✅ Physics calculations (Total_G formula)
- ✅ Tire stress metrics (G² relationship)
- ✅ ROI efficiency scoring
- ✅ Driver skill analysis
- ✅ ML prediction accuracy
- ✅ Edge case handling
- ✅ Multi-race consistency

**Results**: 94.9% pass rate (148/156 tests)

---

## 🧪 Testing

### Run Full Test Suite
```bash
python src/test_analysis.py
```

### Test Output Example
```
================================================================================
  TEST SUMMARY
================================================================================
Total Tests Run: 156
✅ Passed: 148 (94.9%)
❌ Failed: 8 (5.1%)
================================================================================
```

### Key Validations
- ✅ All physics formulas mathematically correct
- ✅ 6,150.23 total tire stress units calculated accurately
- ✅ 682 laps analyzed with perfect aggregation
- ✅ Zero calculation errors in 100+ random sample verifications
- ✅ Both Race 1 and Race 2 data validated

---

## 📊 Sample Results

### Race 1 (Sonoma)
- **Data**: 27.5M raw rows → 450K clean samples
- **Laps**: 682 laps across 30 vehicles
- **Trail Braking**: 13.1% (Amateur level)
- **ROI Distribution**: 33.6% Excellent, 30.5% Good, 36% Wasteful
- **Tire Failure**: Vehicle GR86-002-002 predicted at Lap 22.6

### Race 2 (Sonoma)
- **Data**: 13.6M raw rows → 361K clean samples
- **Laps**: 673 laps across 31 vehicles
- **Trail Braking**: 15.5% (+2.4% improvement!)
- **ROI Distribution**: 3.3% Excellent, 96.7% Wasteful (harder racing!)
- **Tire Failure**: Vehicle GR86-002-002 predicted at Lap 24.1

**Key Insight**: Race 2 shows 92% TERRIBLE efficiency → drivers pushed harder, destroyed tires for minimal time gain. This is exactly what the ROI engine detects!

---

## 🎓 Usage Examples

### Example 1: Quick Analysis
```python
# Load and analyze friction circle
from analysis import load_telemetry_data, clean_telemetry_data, analyze_driver_skill

df = load_telemetry_data('data/Sonoma/Race 1/sonoma_telemetry_R1.csv')
df_clean = clean_telemetry_data(df)
skill = analyze_driver_skill(df_clean)

print(f"Trail Braking: {skill['trail_brake_pct']:.1f}%")
# Output: Trail Braking: 13.1%
```

### Example 2: ROI Analysis
```python
# Calculate ROI efficiency
from roi_engine import calculate_roi_efficiency, predict_tire_failure

lap_roi = calculate_roi_efficiency(lap_summary)
prediction = predict_tire_failure(lap_roi, 'GR86-002-002')

print(f"Predicted failure: Lap {prediction['predicted_failure_lap']:.1f}")
# Output: Predicted failure: Lap 22.6
```

### Example 3: Launch Dashboard
```bash
streamlit run src/streamlit_app.py
# Opens browser at http://localhost:8501
```

---

## 🎨 Dashboard Screenshots

### Overview Tab
- Key metrics: Total laps, vehicles, tire stress, avg G-force
- Driver skill assessment with coaching tips
- ROI distribution summary
- Data quality metrics

### Friction Circle Tab
- Interactive scatter plot of G-forces
- Color-coded by speed (inferno colormap)
- Reference circles at 1.0G and 1.5G
- Shape interpretation (Diamond vs Mushroom)

### ROI Analysis Tab
- Bar chart of efficiency distribution
- Top 10 most efficient laps
- Bottom 10 least efficient laps
- Detailed metrics and recommendations

### Tire Failure Prediction Tab
- ML-powered prediction graph
- Current lap, cumulative stress, stress rate
- Laps remaining to failure
- Pit strategy recommendations

### Vehicle Details Tab
- Tire stress timeline
- AI coaching recommendations
- Detailed lap-by-lap data table
- CSV download capability

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Evina** - *Initial work* - [evinaa21](https://github.com/evinaa21)

---

## 🙏 Acknowledgments

- Toyota/TRD for the hackathon opportunity
- Toyota GR Cup for providing telemetry data
- Streamlit team for the amazing framework
- Open source community for supporting libraries

---

## 📞 Contact

- **GitHub**: [@evinaa21](https://github.com/evinaa21)
- **Project Link**: [https://github.com/evinaa21/Toyota-Racing-ROI](https://github.com/evinaa21/Toyota-Racing-ROI)

---

## 🔮 Future Enhancements

- [ ] GPS track map visualization with speed heatmap
- [ ] Multi-race comparison dashboard
- [ ] Real-time telemetry streaming support
- [ ] Advanced ML models (Random Forest, XGBoost)
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] REST API for third-party integrations
- [ ] Mobile app development
- [ ] Cloud deployment (AWS/Azure/GCP)

---

<div align="center">

**Built with ❤️ for Toyota GR Cup Racing**

🏁 Made for the Toyota/TRD Hackathon 2025 🏁

</div>
