# SafeCrowd Vision 🛡️

Real-time crowd density analysis for public event safety.

**Powered by:** DM-Count · KMeans · XGBoost · DBSCAN · YuNet Face Detection

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python3 -m streamlit run app.py
```

## Deploy with ngrok

To share your app with anyone:

```bash
# Terminal 1: Run the app
python3 -m streamlit run app.py

# Terminal 2: Expose via ngrok
ngrok http 8501
```

Share the ngrok URL with anyone to access your app!

## Tech Stack

- **Frontend**: Streamlit
- **Crowd Counting**: DM-Count (VGG16 encoder-decoder)
- **Face Detection**: YuNet DNN + Haar cascade fallback
- **Clustering**: KMeans, GMM, DBSCAN
- **Classification**: XGBoost
- **RL Agent**: DQN for evacuation policy learning
