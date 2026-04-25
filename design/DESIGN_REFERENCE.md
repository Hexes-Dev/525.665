# Design Reference: recursive IMU Position Estimation Network

## 1. Overview
The goal is to replace a traditional Extended Kalman Filter (EKF) for sensor fusion with an auto-regressive recursive neural network. The model will fuse data from six co-located IMUs to predict the displacement in the North-East-Down (NED) frame relative to a starting point.

## 2. Data Engineering & Transformation
### Coordinate Frame
*   **Target**: Absolute GPS coordinates (Lat/Lon/Alt) are converted into a local tangent plane (NED).
*   **Conversion**: Use `latlon_to_ned(current, origin)` to calculate offsets in meters.
*   **Labels**: The network is trained against $\Delta \text{Position}_{\text{NED}}$ between consecutive timesteps.

### Normalization
*   **Method**: Z-score normalization ($\frac{x - \mu}{\sigma}$).
*   **Process**: A pre-computation script samples the calibrated dataset to generate `src/ml/norm_params.json`. These statistics are applied in the data loader to ensure consistent scaling of raw sensor values.

## 3. Smart Data Loader (`src/ml/imu_data_loader.py`)
To handle large datasets and minimize DB query overhead, the loader uses an anchor-based sliding window approach:
*   **Zero-Velocity Anchors**: Query GPS data to identify timestamps where `speed < 0.01 m/s`. These serve as sequence starting points ($t_0$).
*   **Dataset Splitting**: To ensure no temporal leakage, the dataset is split at the anchor level. Anchors are shuffled and partitioned into training and testing sets based on a configurable percentage before initializing separate `IMUDataset` instances.
*   **Dynamic Windowing**: For each anchor, a configurable duration (initially 10s) of IMU data is queried: $[t_0, t_0 + \text{duration}]$.
*   **Input Vector (Dimension 17)**:
    *   `Sensor Data (10)`: `[acc(3), gyr(3), mag(3), temp(1)]`
    *   `Source ID (6)`: One-hot encoding for the six IMU sources.
    *   `Temporal Delta (1)`: $\Delta t$ since the previous reading in the aggregated stream.

## 4. Recursive Network Architecture (`src/ml/imu_model.py`)
The model uses an auto-regressive structure to maintain a persistent internal state of position estimation.
*   **Recurrent Core**: A GRU (Gated Recurrent Unit) processes the sequence of sensor readings.
*   **Recursive Feedback Loop**:
    1.  The GRU hidden state $h_t$ is passed through a linear **Bottleneck Layer** to produce a feedback vector $f_t$.
    2.  This vector $f_t$ is concatenated with the input at timestep $t+1$: $\text{Input}_{t+1} = \text{concat}(\text{SensorData}_{t+1}, f_t)$.
    3.  This creates an explicit feedback loop separate from the GRU's internal gates, allowing for more stable state tracking.
*   **Output Head**: A linear layer mapping the final processed state to a 3D displacement vector ($\Delta \text{North, } \Delta \text{East, } \Delta \text{Down}$).

## 5. Execution & Validation Pipeline
1.  **Parameters Generation**: Compute and save normalization stats; verify NED conversion accuracy.
2.  **Data Pipeline implementation**: Implement anchor-based loading and target alignment.
3.  **Model Development**: Build the recursive GRU with auto-regressive feedback and verify tensor shapes.
4.  **Iterative Training**: Start with 10s sequences, then incrementally increase window duration as model stability improves.
