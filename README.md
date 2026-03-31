🌟 Project Overview
Traffic congestion is a major problem in modern cities. Predicting future traffic conditions helps:

🗺️ Navigation apps (Google Maps, Waze) suggest better routes
🚦 Smart city traffic signal management
🚑 Emergency services find fastest routes
🚌 Public transport schedule optimization

This project implements and compares three deep learning models for traffic speed prediction:

GCN — Graph Convolutional Network (spatial only)
T-GCN — Temporal GCN (spatial + temporal combined)
GRU — Gated Recurrent Unit (temporal only)


🏗️ Architecture
T-GCN Architecture
Traffic Sensor Data (Road Network Graph)
            ↓
    ┌───────────────┐
    │  GCN Layer    │  ← Learns spatial patterns (road connections)
    │  (Graph Conv) │
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │  GRU Layer    │  ← Learns temporal patterns (time changes)
    │  (Recurrent)  │
    └───────┬───────┘
            ↓
    ┌───────────────┐
    │ Linear Layer  │  ← Output predicted traffic speeds
    └───────────────┘
            ↓
    Predicted Traffic Speed (next 3 time steps)
How It Works

GCN treats road intersections as nodes and roads as edges in a graph
GRU remembers past traffic patterns to predict future ones
T-GCN combines both — understanding road network structure AND time patterns simultaneously


📊 Datasets
DatasetCityCountrySensorsTypeTime IntervalLos-LoopLos AngelesUSA 🇺🇸207Highway Traffic5 minutesShenzhenShenzhenChina 🇨🇳156Urban Road Traffic5 minutes
Data Configuration

Input (seq_len): Last 12 time steps (past 60 minutes)
Output (pre_len): Next 3 time steps (future 15 minutes)
Train/Test Split: 80% training / 20% testing


🤖 Models
1. GCN (Graph Convolutional Network)

Parameters: 963
Captures: Spatial patterns only
Uses adjacency matrix of road network

2. T-GCN (Temporal Graph Convolutional Network)

Parameters: 12,900
Captures: Spatial + Temporal patterns
Combines GCN inside GRU cell

3. GRU (Gated Recurrent Unit)

Parameters: 12,900
Captures: Temporal patterns only
Best performing model in experiments


📈 Results
Los-Loop Dataset (Los Angeles, USA)
ModelEpochsAccuracyMAERMSER2 ScoreGCN100085.69%5.968.400.633GCN2585.37%6.058.590.616T-GCN35787.54%5.087.310.722T-GCN5087.20%5.297.510.706GRU5090.18% 🏆3.28 🏆5.76 🏆0.827 🏆
Shenzhen Dataset (China)
ModelEpochsAccuracyMAERMSER2 ScoreGCN2558.18%4.486.000.669T-GCN2557.75%4.616.060.663GRU2570.56% 🏆2.90 🏆4.22 🏆0.836 🏆
🏆 Key Finding
GRU is the best model on BOTH datasets — achieving 90.18% accuracy on Los-Loop and 70.56% on Shenzhen with only 50 epochs of training.

📁 Project Structure
T-GCN-master/
├── T-GCN/
│   ├── T-GCN-PyTorch/          # Main PyTorch implementation
│   │   ├── data/               # Dataset files (Los-Loop, Shenzhen)
│   │   ├── models/             # Model definitions
│   │   │   ├── gcn.py          # GCN model
│   │   │   ├── gru.py          # GRU model
│   │   │   └── tgcn.py         # T-GCN model (GCN + GRU)
│   │   ├── tasks/              # Training task definitions
│   │   ├── utils/              # Helper functions
│   │   │   └── graph_conv.py   # Graph convolution utilities
│   │   ├── main.py             # Main entry point
│   │   ├── requirements.txt    # Dependencies
│   │   └── lightning_logs/     # Training logs and checkpoints
│   └── T-GCN-TensorFlow/       # TensorFlow implementation
├── A3T-GCN/                    # Attention T-GCN variant
├── AST-GCN/                    # Semantic T-GCN variant
├── HoT-GCN/                    # Higher-order T-GCN
├── STGCNN/                     # Spatio-temporal GCN
├── data/                       # Shared datasets
├── big picture.png             # Architecture diagram
└── README.md                   # This file

⚙️ Installation
Prerequisites

Python 3.8
Windows / Linux / Mac

Step 1 — Clone the repository
bashgit clone https://github.com/Tanweer786-stark/T-GCN-Traffic-Prediction.git
cd T-GCN-Traffic-Prediction
Step 2 — Navigate to PyTorch implementation
bashcd T-GCN-master/T-GCN/T-GCN-PyTorch
Step 3 — Install dependencies
bashpython -m pip install -r requirements.txt --user
python -m pip install scipy --user
python -m pip install pytorch-lightning==1.9.5 --user
python -m pip install tensorboard --user

🚀 How to Run
Run GCN Model
bash# Quick test (25 epochs)
python main.py --model_name GCN --max_epochs 25

# Full training (1000 epochs)
python main.py --model_name GCN
Run T-GCN Model
bashpython main.py --model_name TGCN --max_epochs 50
Run GRU Model (Best Performance)
bashpython main.py --model_name GRU --max_epochs 50
Run on Shenzhen Dataset
bashpython main.py --model_name GRU --data shenzhen --max_epochs 25
All Available Parameters
bashpython main.py \
  --model_name GRU \        # GCN, TGCN, or GRU
  --data losloop \          # losloop or shenzhen
  --max_epochs 50 \         # number of training epochs
  --hidden_dim 64 \         # hidden layer dimension
  --learning_rate 0.001 \   # learning rate
  --batch_size 32 \         # batch size
  --seq_len 12 \            # input sequence length
  --pre_len 3               # prediction length

📊 TensorBoard Visualization
View training graphs including loss, accuracy, MAE, and RMSE:
bash# Start TensorBoard
python -m tensorboard.main --logdir=lightning_logs

# Open in browser
http://localhost:6006

📦 Requirements
numpy
matplotlib
pandas
torch
pytorch-lightning==1.9.5
torchmetrics
python-dotenv
scipy
tensorboard

🔮 Future Work

 Add GPU support for faster training
 Include weather data as additional features
 Implement attention mechanism (A3T-GCN)
 Test on more cities and datasets
 Increase hidden_dim for better accuracy
 Deploy as REST API for real-time predictions
 Add accident and event data as features


📊 Model Comparison Summary
Accuracy Comparison (Los-Loop Dataset):
GCN   ████████████████████████████████████████  85.69%
TGCN  ██████████████████████████████████████████ 87.54%
GRU   ████████████████████████████████████████████ 90.18% 🏆

Accuracy Comparison (Shenzhen Dataset):
GCN   ████████████████████████████  58.18%
TGCN  ███████████████████████████   57.75%
GRU   ███████████████████████████████████  70.56% 🏆

