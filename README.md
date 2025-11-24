# Toyota Racing ROI Engine 🏎️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75)

> **"In racing, speed is easy. Efficiency is hard."**

The **Toyota Racing ROI Engine** is a financial modeling tool for race strategy. Instead of dollars, we track **Tire Life** as the currency. It calculates the "Cost per Corner" for every lap, helping Race Engineers identify when a driver is "overspending" tire grip for minimal time gain.

---

## 🚨 The Problem
Drivers often destroy their tires blindly. They push 100% on every lap, overheating the rubber and falling off the "cliff" of performance late in the race.
*   **The Cost:** A 0.1s gain on Lap 5 might cost 2.0s of pace on Lap 20 due to degradation.
*   **The Gap:** Traditional telemetry shows *what* happened (G-Force, Speed), but not *what it cost* (Tire Energy).

## 💡 The Solution
We treat tire life like a bank account.
*   **Investment:** Tire Stress (Work Done = $G^2 \times Speed \times Distance$)
*   **Return:** Lap Time Improvement
*   **ROI:** Time Gained per Unit of Tire Energy Spent.

---

## ✨ Key Features

### 1. Physics-Based "Work Done" Analysis
We don't just look at G-Force peaks. We calculate the **Total Energy (Joules)** put into the tire carcass using a physics model that accounts for:
*   **Combined G-Loading:** $\sqrt{Lat^2 + Long^2}$
*   **Vehicle Speed:** Higher speeds = exponentially more energy.
*   **Corner Duration:** Long sweepers kill tires faster than sharp hairpins.

### 2. Machine Learning Pit Strategy
A **Linear Regression Model** tracks the cumulative stress on the tires and predicts the exact lap of failure.
*   **Adaptive Thresholds:** The model learns from the driver's specific aggression level.
*   **"BOX BOX BOX" Alerts:** Real-time warnings when tire failure is imminent within the next 5 laps.

### 3. Interactive "Pit Wall" Dashboard
A Streamlit-based command center for Race Engineers:
*   **ROI Heatmap:** Instantly spot "Wasteful" laps (Red) vs. "Efficient" laps (Green).
*   **Friction Circle Analysis:** Visualizes the driver's usage of available grip.
*   **Coaching Insights:** Automated text recommendations (e.g., *"Lap 7 was WASTEFUL: +15% Stress for -0.02s Gain"*).

---

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Toyota-Racing-ROI.git
    cd Toyota-Racing-ROI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard:**
    ```bash
    streamlit run src/streamlit_app.py
    ```

4.  **Upload Data:**
    *   Use the sidebar to upload your `.csv` telemetry files.
    *   Or select the pre-loaded `Sonoma Race 1` dataset.

---

## 🛠️ Tech Stack
*   **Backend:** Python, Pandas, NumPy, SciPy (Signal Processing)
*   **Machine Learning:** Scikit-Learn (Linear Regression)
*   **Frontend:** Streamlit
*   **Visualization:** Plotly Interactive Charts, Matplotlib

---

## 🏆 Hackathon Context
Built for the **Toyota GR Cup Hackathon 2025**.
*   **Challenge:** Optimize race strategy using telemetry data.
*   **Innovation:** Applying financial ROI principles to physics data.

---

*Engineered for Speed. Optimized for Victory.* 🏁
