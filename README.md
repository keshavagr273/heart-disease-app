# Heart Disease Prediction App

## Overview
This project is a web application that predicts the likelihood of heart disease based on user-provided health parameters. It leverages a trained machine learning model to provide instant predictions through a user-friendly interface.

## Features
- Predicts heart disease risk using a trained neural network model (`heart_disease_mode1l.h5`).
- Simple web interface for inputting health data.
- Real-time prediction results.
- Clean and responsive UI with custom CSS.

## Project Structure
```
heart_app/
│
├── app.py                  # Main Flask application
├── heart_disease_mode1l.h5 # Trained Keras model
├── heart_predict.ipynb     # Jupyter notebook for model training/experiments
├── requirements.txt        # Python dependencies
├── static/
│   └── style.css           # Custom CSS for the web app
├── templates/
│   └── index.html          # Main HTML template
└── README.md               # Project documentation
```

## Model and Techniques Used

### Dataset
- The model was trained on the "Heart Disease Health Indicators" dataset (BRFSS 2015), containing 253,680 samples and 22 features related to health and lifestyle.
- The target variable is `HeartDiseaseorAttack` (binary: 1 for heart disease/attack, 0 for none).

### Data Preprocessing
- **Feature Selection:**
  - Feature importance was evaluated using a Random Forest classifier.
  - The most important features were retained: `BMI`, `Fruits`, `GenHlth`, `MentHlth`, `PhysHlth`, `Age`, `Education`, and `Income`.
- **Balancing:**
  - The dataset was imbalanced; SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the classes.
- **Scaling:**
  - Features were standardized using `StandardScaler` for improved neural network performance.

### Model Architecture
- The model is a Keras Sequential neural network with the following layers:
  - Input layer matching the number of selected features.
  - Dense layer with 128 units, ReLU activation, He normal initialization, and Dropout (0.15).
  - Dense layer with 64 units, ReLU activation, He normal initialization, and Dropout (0.15).
  - Dense layer with 32 units, ReLU activation, He normal initialization, L2 regularization.
  - Output layer with 1 unit and sigmoid activation (for binary classification).
- **Optimizer:** Adam (learning rate 0.0003)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, Precision, Recall
- **Callbacks:** EarlyStopping (patience=15), ReduceLROnPlateau (patience=5)

### Training and Evaluation
- The model was trained for up to 35 epochs with a batch size of 64, using 20% of the data for validation.
- Training progress was monitored using loss and accuracy metrics.
- The final model was saved as `heart_disease_mode1l.h5`.

### Prediction Pipeline
- The web app collects user input for the selected features and preprocesses them as required.
- The trained model predicts the probability of heart disease, and the app displays a risk assessment ("High Risk" or "Low Risk").

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or suggestions, please contact:
- **Keshav Agrawal**: <keshavagrawal273@gmail.com> 
