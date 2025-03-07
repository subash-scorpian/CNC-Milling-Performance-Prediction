Problem Statement:
To analyze CNC milling operations and build predictive models for tool wear detection,
inadequate clamping detection, and process completion prediction using sensor data collected from machining experiments.
Business Use Cases:
•	Predictive Maintenance: Identify tool wear and schedule timely tool replacements to avoid production interruptions.
•	Quality Assurance: Detect insufficient clamping that can lead to product defects.
•	Process Optimization: Optimize machining parameters to minimize cycle times and reduce operational costs.
•	Operational Safety: Monitor machining parameters to prevent unsafe machining conditions.
Approach:
1.	Data Understanding: Analyze the structure of train.csv and the time series files.
2.	Data Preprocessing: Handle missing values and anomalous readings.
3.	Feature Engineering: Extract statistical, temporal, and frequency-based features.
4.	Model Development: DL Train supervised models for classification tasks.
5.	Evaluation and Optimization: Evaluate models using appropriate metrics and fine-tune as needed.
6.	Deployment: Develop a prototype model for real-time predictions.
Results: 
•	Accurate detection of worn and unworn tools.
•	Reliable prediction of machining completion success.
•	Early detection of inadequate clamping to reduce defective parts.
Project Evaluation metrics:
•	Model Execution Time for Real-Time Application
Technical Tags:
•	Python, Pandas, NumPy, Scikit-Learn
•	Time Series Analysis
•	Deep Learning(Binary Classification)
•	Predictive Maintenance
•	CNC Machining Data
Data Set:
•	train.csv: Experiment-level summary
•	experiment_01.csv to experiment_18.csv: Time series data
Format: CSV files
Key Variables:
•	Experiment-level: tool_condition, clamp_pressure, feed_rate, machining_completed, passed_visual_inspection
•	Time Series: Motor performance variables for X, Y, Z axes and spindle, including position, velocity, acceleration, current, voltage, and power.
Data Set Explanation:
•	Each experiment corresponds to machining a wax block with an "S" shape.
•	Time series data sampled every 100ms for four motors: X, Y, Z, and spindle.
•	Labels available for tool condition (worn/unworn), process completion, and visual inspection.
Preprocessing Steps:
•	Handle anomalous readings (e.g., M1_CURRENT_FEEDRATE = 50 or X1_ActualPosition = 198).
•	Normalize features for better model performance.
•	Aggregate time series data using statistical or frequency domain techniques.

The dataset can be used in classification studies such as:

(1) Tool wear detection
Supervised binary classification could be performed for identification of worn and unworn cutting tools. Eight experiments were run with an unworn tool while ten were run with a worn tool (see tool_condition column for indication).

(2) Detection of inadequate clamping
The data could be used to detect when a workpiece is not being held in the vise with sufficient pressure to pass visual inspection (see passed_visual_inspection column for indication of visual flaws). Experiments were run with pressures of 2.5, 3.0, and 4.0 bar. The data could also be used for detecting when conditions are critical enough to prevent the machining operation from completing (see machining_completed column for indication of when machining was preemptively stopped due to safety concerns).


General data from a total of 18 different experiments are given in train.csv and includes:

Inputs (features)

No : experiment number
material : wax
feed_rate : relative velocity of the cutting tool along the workpiece (mm/s)
clamp_pressure : pressure used to hold the workpiece in the vise (bar)

Outputs (predictions)

tool_condition : label for unworn and worn tools
machining_completed : indicator for if machining was completed without the workpiece moving out of the pneumatic vise
passed_visual_inspection: indicator for if the workpiece passed visual inspection, only available for experiments where machining was completed


Time series data was collected from 18 experiments with a sampling rate of 100 ms and are separately reported in files experiment_01.csv to experiment_18.csv. Each file has measurements from the 4 motors in the CNC (X, Y, Z axes and spindle). These CNC measurements can be used in two ways:

(1) Taking every CNC measurement as an independent observation where the operation being performed is given in the Machining_Process column. Active machining operations are labeled as "Layer 1 Up", "Layer 1 Down", "Layer 2 Up", "Layer 2 Down", "Layer 3 Up", and "Layer 3 Down". 

(2) Taking each one of the 18 experiments (the entire time series) as an observation for time series classification


The features available in the machining datasets are:

X1_ActualPosition: actual x position of part (mm)
X1_ActualVelocity: actual x velocity of part (mm/s)
X1_ActualAcceleration: actual x acceleration of part (mm/s/s)
X1_CommandPosition: reference x position of part (mm)
X1_CommandVelocity: reference x velocity of part (mm/s)
X1_CommandAcceleration: reference x acceleration of part (mm/s/s)
X1_CurrentFeedback: current (A)
X1_DCBusVoltage: voltage (V)
X1_OutputCurrent: current (A)
X1_OutputVoltage: voltage (V)
X1_OutputPower: power (kW)

Y1_ActualPosition: actual y position of part (mm)
Y1_ActualVelocity: actual y velocity of part (mm/s)
Y1_ActualAcceleration: actual y acceleration of part (mm/s/s)
Y1_CommandPosition: reference y position of part (mm)
Y1_CommandVelocity: reference y velocity of part (mm/s)
Y1_CommandAcceleration: reference y acceleration of part (mm/s/s)
Y1_CurrentFeedback: current (A)
Y1_DCBusVoltage: voltage (V)
Y1_OutputCurrent: current (A)
Y1_OutputVoltage: voltage (V)
Y1_OutputPower: power (kW)

Z1_ActualPosition: actual z position of part (mm)
Z1_ActualVelocity: actual z velocity of part (mm/s)
Z1_ActualAcceleration: actual z acceleration of part (mm/s/s)
Z1_CommandPosition: reference z position of part (mm)
Z1_CommandVelocity: reference z velocity of part (mm/s)
Z1_CommandAcceleration: reference z acceleration of part (mm/s/s)
Z1_CurrentFeedback: current (A)
Z1_DCBusVoltage: voltage (V)
Z1_OutputCurrent: current (A)
Z1_OutputVoltage: voltage (V)

S1_ActualPosition: actual position of spindle (mm)
S1_ActualVelocity: actual velocity of spindle (mm/s)
S1_ActualAcceleration: actual acceleration of spindle (mm/s/s)
S1_CommandPosition: reference position of spindle (mm)
S1_CommandVelocity: reference velocity of spindle (mm/s)
S1_CommandAcceleration: reference acceleration of spindle (mm/s/s)
S1_CurrentFeedback: current (A)
S1_DCBusVoltage: voltage (V)
S1_OutputCurrent: current (A)
S1_OutputVoltage: voltage (V)
S1_OutputPower: current (A)
S1_SystemInertia: torque inertia (kg*m^2)

M1_CURRENT_PROGRAM_NUMBER: number the program is listed under on the CNC
M1_sequence_number: line of G-code being executed
M1_CURRENT_FEEDRATE: instantaneous feed rate of spindle

Machining_Process: the current machining stage being performed. Includes preparation, tracing up  and down the "S" curve involving different layers, 
and repositioning of the spindle as it moves through the air to a certain starting point

Summary
Model Construction:
The code builds a deep learning model using LSTM layers with dropout regularization to process a sequence with one timestep and 16 features.

Multi-output Setup:
It branches out into three separate dense layers, each predicting a binary outcome (using sigmoid activation) for different tasks.

Compilation and Training:
The model is compiled with the Adam optimizer and binary crossentropy loss for each output, then trained using a train-test split.

This multi-output architecture is useful when you need to predict multiple related outputs from the same input data, all while sharing the same learned features from the LSTM layers.
