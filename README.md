# 🇪🇺 Europe CyberScope — CSOMA (CyberSecurity Job Market Analyzer)

**CSOMA (CyberSecurity Job Market Analyzer)** is a data pipeline and analytics platform designed to **collect, process, and visualize cybersecurity job advertisements across Europe in (simulated) real time**.  
The system leverages open-source technologies to identify **regional trends**, **in-demand skills**, and **evolving market dynamics** in the cybersecurity sector.

---

## Stream mining Architecture

```
CSV File → Kafka Producer → Kafka Topic → Kafka Consumer → Cassandra  → Streamlit Live Dashboard.
              ↓                 ↓                            ↓
        (simulates      (KRaft mode -               (linkedin_jobs)
         real-time)     no ZooKeeper)                   (ecsf)
         
         PARALLEL EXECUTION: Producer & Consumer run simultaneously
```
Its technical design follows the streaming-pipeline principles described by Narkhede et al. (2017), using Apache Kafka for real-time ingestion and Apache Cassandra for scalable, reliable data storage.

---

## ⚙️ Quick Start Guide

### Clone the Repository
```bash
git clone https://github.com/paulinaeb/ost-sm-project.git
cd ost-sm-project
```
### Automated Workflow (Recommended)

Use these one-liners to get up and streaming fast:

```bash
# Linux/macOS/WSL
bash deploy.sh
```

```powershell
# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\deploy.ps1
```

One-line alternative (does the same in a fresh session):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\deploy.ps1
```
Notes:
- `Set-ExecutionPolicy -Scope Process` only applies to the current PowerShell window and is temporary.
- You only need to run it once per session before calling `.\deploy.ps1` (subsequent runs can omit it).
- If your policy already allows script execution, skip it entirely and just run `.\deploy.ps1`.

Then, to reset and restart streaming:

```bash
# Linux/macOS/WSL
bash restart_simulation.sh
```

```powershell
# Windows PowerShell
.\restart_simulation.ps1
```
If this is a brand new PowerShell session and you have not set the execution policy yet, you can chain it:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\restart_simulation.ps1
```
Otherwise just use `.\restart_simulation.ps1` directly.

### For more details and an alternative option check streaming folders README

---

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit Dashboard | http://localhost:8501 | See stream mining & visualizations|
| Cassandra Web UI | http://localhost:8081 | Browse Cassandra data |
| Kafka UI | http://localhost:8080 | Monitor Kafka topics |
| Cassandra CQL | localhost:9042 | Direct CQL access |
| Kafka Broker | localhost:29092 | Producer/Consumer connection |

---

## 📊 Dashboard & Visualizations

The **Streamlit Dashboard** provides an interactive interface to explore cybersecurity job market data across Europe. 
Access it at http://localhost:8501 after starting the services.

### Navigation Tabs

The dashboard features horizontal navigation with the following sections:

| Tab | Description | Author |
|-----|-------------|--------|
| **📈 Dashboard** | Real-time streaming overview with live job ingestion monitoring, time-based aggregations, and quick statistics | Ahad |
| **🌍 Country Radar** | European job market analysis with interactive visualizations | Paulina |
| **📈 Predictive Insights** | Market trend forecasting and predictions | Tibor |
| **🔍 Matching Tracker** | Job-skill matching and recommendation system | Sameha |
| **📡 Change Detector** | Real-time anomaly detection and market shifts | Ahad |

The Country Radar builds on insights from Ogryzek & Jaskulski (2025), whose GIS-based choropleth mapping shows how spatial visualisation can clarify regional labour-market patterns.

The Predictive Insights module builds on Cerqueira et al. (2019) and Chen & Guestrin (2016), applying recursive multi‑step forecasting with tree‑based models (Random Forest/XGBoost) to capture non‑linear temporal dynamics and enhance short‑horizon job‑market predictions.

All visualizations support dual modes:
- **Database Mode**: Historical data analysis
- **Streaming Mode**: Real-time updates with N-second refresh (auto-refresh enabled)

---

## 📁 Dataset Sources

| Type | Location | Description |
|------|-----------|--------------|
| Dynamic (simulated) | [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1Ult_m13_--7MYIEA8JGtRRzqX8hyaz3W) | Periodically updated simulated job ads |
| Static | [GitHub Dataset – ENISA ECSF](https://github.com/opliyal3/ENISA-ECSF-Dataset/tree/main) | Reference dataset for ECSF-aligned skills |

---

## 👥 Team Contributions

| Name | Nationality | Role & Responsibilities |
|------|-------------|------------------------|
| **Nasser Samiha** 🇸🇾 | Syrian | **Data Preprocessing & Matching Tracker**<br/>• Cleaned ECSF and LinkedIn datasets and stored ECSF in Cassandra<br/>• Built fuzzy matching pipeline for job-skill alignment<br/>• Developed ECSF/Jobs matching visualization dashboard |
| **Espejo Paulina** 🇻🇪 | Venezuelan | **Stream Mining & Country Radar**<br/>• Implemented Kafka producer/consumer for real-time job simulation and its storage to Cassandra<br/>• Designed European job market geographic visualization dashboard<br/>• Initiated containerization with Docker compose
| **Ahad Rezaul Khan** 🇧🇩 | Bangladesh | **Real-time Dashboard & Change Detection**<br/>• Created live streaming dashboard with auto-refresh<br/>• Implemented batch analytics and forecasting models<br/>• Built anomaly detection for market shifts |
| **Buti Tibor** 🇭🇺 | Hungarian | **Deployment & Predictive Insights**<br/>• Created comprehensive pipeline for deployment <br/>• Developed time-series forecasting dashboard |

---

## 👩‍💻 Maintainers

**Europe CyberScope Team**  
Contributions and issue reports are welcome — please open a GitHub issue or submit a pull request.


**Reference:**  
1. (ISC)². (2023). Cybersecurity Workforce Study: How the Economy, Skills
Gap and Artificial Intelligence are Challenging the Global Workforce.
https://www.isc2.org/research
2. Furnell, S. (2021). "The cybersecurity workforce and skills," Computers Security,
100, 102080 https://doi.org/https://doi.org/10.1016/j.cose.2020.102080
3. National Institute of Standards and Technology (NIST). (2020). NICE Workforce
Framework for Cybersecurity (NIST Special Publication 800-181, Rev. 1). U.S.
Department of Commerce. https://doi.org/https://doi.org/10.6028/NIST.SP.800-
181r1
4. ENISA. (2022). European Cybersecurity Skills Framework (ECSF). European
Union Agency for Cybersecurity. https://www.enisa.europa.eu/topics/skills-and-
competences/skills-development/european-cybersecurity-skills-framework-ecsf
5. Lakshman, A., and Malik, P. (2010). Cassandra: A Decentralized Structured Storage
System. ACM SIGOPS Operating Systems Review, 44(2), 35–40.
6. Treuille, A., and Nielsen, A. (2020). Streamlit: A Framework for Machine Learning
and Data Science Web Apps. Streamlit Inc. Available at: https://streamlit.io .
7. Snow, A., Gillies, R., and Fan, A. (2022). Building Interactive Data Applications
with Streamlit. Proceedings of the Python Web Conference. Streamlit Inc
8. Landauer, M., Skopik, F., Stojanović, B. et al. A review of time-series analysis for
cyber security analytics: from intrusion detection to attack prediction. Int. J. Inf.
Secur. 24, 3 (2025). https://doi.org/https://doi.org/10.1007/s10207-024-00921-0
9. Andrea, M., Ana, K. , Karlo, K. , Mattea, M. Insights at a
Glance: Unravelling Spatial Trends with Bivariate Choropleth Mapping
https://doi.org/https://doi.org/10.5194/ica-abs-7-105-2024
10. Mrini, K., Sharma, R., Dillenbourg, P. (2017). Detecting trends
in job advertisements. École Polytechnique Fédérale de Lausanne.
https://infoscience.epfl.ch/record/256472
11. Narkhede, N., Shapira, G., Palino, T. (2017). Kafka: The definitive guide: Real-
time data and stream processing at scale. O’Reilly Media.
12. Ogryzek, M., Jaskulski, M. (2025). Applying methods of exploratory data analysis
and methods of modeling the unemployment rate in spatial terms in Poland. Applied
Sciences, 15(8), 4136. https://doi.org/https://doi.org/10.3390/app15084136