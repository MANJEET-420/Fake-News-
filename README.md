# Fake News Detection System

## About the Project

The Fake News Detection System is a Machine Learning project that predicts whether a news article or headline is fake or real based on its text.

The application uses a trained Machine Learning model and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to process news text and generate predictions through a Streamlit web interface.

## Features

- Enter news headlines or article text.
- Convert text into TF-IDF features.
- Predict whether news is fake or real.
- Simple and interactive Streamlit interface.
- Load a pre-trained model and vectorizer from pickle files.

## Technologies Used

- Python
- Machine Learning
- Scikit-learn
- TF-IDF Vectorizer
- Streamlit
- Pandas
- NumPy
- Pickle

## Project Structure

```text
Fake-News-Detection/
├── artifacts/
│   ├── model.pkl
│   └── vectorizer.pkl
├── app.py
├── requirements.txt
├── train.csv
├── README.md
├── .gitignore
└── Fake News Classification.ipynb
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Fake-News-Detection.git
cd Fake-News-Detection
```

Replace `YOUR-USERNAME` with your GitHub username and adjust the repository name if necessary.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## Model and Prediction

The application loads the following files:

- `artifacts/model.pkl` — trained Machine Learning model.
- `artifacts/vectorizer.pkl` — fitted TF-IDF vectorizer.

The input text is transformed using the saved vectorizer before being passed to the trained model.

**Important:** The fake/real prediction labels must match the labels used during model training.

## Dataset

The project includes `train.csv` for model training or experimentation. Update this section with the original dataset source and attribution when known.

## Limitations

- Predictions depend on the quality and coverage of the training data.
- The model may misclassify genuine or misleading news.
- This application does not independently verify news against authoritative sources.
- Predictions should not be treated as definitive proof that a news article is true or false.

## Future Improvements

- Evaluate the model using accuracy, precision, recall and F1-score.
- Add a confusion matrix and model evaluation report.
- Improve the interface and display prediction confidence when supported by the model.
- Deploy the application online.

## Author

Manjeet Kumar

## License

Add a suitable open-source license if you intend to distribute this project publicly.
