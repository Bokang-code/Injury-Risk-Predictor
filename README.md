# Athlete Injury Risk Predictor

A machine-learning project exploring whether player-level characteristics and training-load indicators can help identify potential injury risk in athletes. The project combines model experimentation, a Streamlit prediction interface and Power BI visualisations.

The work compares Decision Tree, Random Forest and K-Means approaches, with a Random Forest classifier selected as the main predictive model in the project. The repository also includes an interactive dashboard application and supporting academic analysis.

## Project overview

The application collects player and training information, passes the features to a saved Random Forest model and displays a risk category. Inputs include:

- Gender, body weight and playing position
- Training intensity and weekly training hours
- Recovery days, match frequency and rest between events
- Fatigue, performance and team-contribution scores

The repository also contains a notebook and Power BI dashboard for exploratory analysis and communicating patterns in the dataset.

## Technology

- Python
- Pandas and NumPy
- scikit-learn / Random Forest
- Joblib for model persistence
- Streamlit for the interactive application
- Jupyter Notebook for model development
- Power BI for dashboarding

## Repository structure

```text
.
└── Injury_Predictor/
    ├── app.py                              # Streamlit interface
    ├── RFmodel.pkl                         # Saved Random Forest model
    ├── model.pkl                           # Additional saved model artefact
    ├── UGProject-Notebook.ipynb             # Project notebook
    ├── UG_Proejct_Random_Forest-checkpoint.ipynb
    ├── SMOTE_collegiate_athlete_injury_dataset.csv.arff.csv
    └── UG_DashBoard.pbix                   # Power BI dashboard
```

## Run locally

Clone the repository:

```bash
git clone https://github.com/Bokang-code/Injury-Risk-Predictor.git
cd Injury-Risk-Predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the application dependencies:

```bash
pip install streamlit numpy pandas scikit-learn joblib
```

Launch the application from the folder containing `app.py` and `RFmodel.pkl`:

```bash
cd Injury_Predictor
streamlit run app.py
```

## How the prediction works

The app converts the selected categorical inputs into numeric values, builds a feature row and calls `predict_proba` on the saved model. It then applies additional rule-based adjustments for high fatigue, limited recovery days and high weekly training hours before displaying one of four risk bands.

Because the displayed percentage includes these manually added adjustments, it should not be interpreted as a calibrated probability from the Random Forest model.

## Important limitations

**This project is an academic prototype, not a clinically validated injury-screening tool.**

- The model output is not medical advice and must not be used to diagnose, clear or restrict an athlete.
- Predictive performance depends on the dataset, feature definitions, sampling and validation methodology.
- The app's extra rule-based percentage adjustments are heuristic and may distort the model's probability estimate.
- The saved model must be paired with exactly the same feature order and preprocessing used during training.
- Risk bands are application-defined thresholds and should not be treated as established clinical cut-offs.

## Recommended next steps

- Document dataset provenance, feature definitions and target-label construction.
- Evaluate with stratified cross-validation and a held-out test set.
- Report sensitivity/recall, specificity, precision, F1-score, ROC-AUC, calibration and a confusion matrix.
- Check for athlete-level or time-based leakage between training and test data.
- Persist preprocessing together with the estimator in a single reproducible pipeline.
- Calibrate predicted probabilities and validate thresholds with domain experts before any real-world use.
- Add input validation, automated tests and model-version metadata.

---

**Author:** [Bokang Khasipe](https://github.com/Bokang-code)
