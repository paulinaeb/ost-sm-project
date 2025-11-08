# Europe CyberScope - CSOMA (CyberSecurity Job Market Analyzer)
The Cybersecurity Job Market Analyzer (CSOMA) aims to acquire, store, and analyze cybersecurity job advertisements across Europe in real time. The system uses open-source technologies to process and visualize job data, helping identify regional trends and emerging skill demands.

# Dataset sources 
datasets/dynamic (simulated)
https://drive.google.com/drive/u/1/folders/1Ult_m13_--7MYIEA8JGtRRzqX8hyaz3W
datasets/static
https://github.com/opliyal3/ENISA-ECSF-Dataset/tree/main

# How to run: get cassandra config and data!

0. Clone this repo 
git clone <your-repo-url>
cd ost-sm-project

1. Start containers
docker-compose up -d

-- Wait for Cassandra to be healthy (~1-2 minutes)
docker logs -f cassandra-dev

2. Create venv and install python dependencies for loading data
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install requirements.txt

3. Load data with python
python preprocessing/ECSF/load_ecsf.py