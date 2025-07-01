🫁 Lung Cancer Risk Predictor
=============================

A web application that predicts the risk of lung cancer (Low, Medium, or High) based on user-provided health and environmental factors. It leverages a Logistic Regression model built with scikit-learn, a Flask backend, Tailwind CSS for a responsive UI, and the Hugging Face Inference API for AI-generated explanations.

🚀 Features
-----------

*   **User-Friendly Interface**: Input health and environmental factors via a form with Yes/No radio buttons, Age (numeric), and Gender (dropdown).
    
*   **Machine Learning Model**: Logistic Regression model trained on a dataset to predict lung cancer risk.
    
*   **AI-Powered Explanations**: Hugging Face's Mixtral-8x7B model generates concise, friendly explanations of predicted risk levels with practical lifestyle suggestions.
    
*   **Responsive UI**: Tailwind CSS ensures a clean, modern design across devices.
    
*   **Scalable Backend**: Flask handles predictions and API integration efficiently.
    
*   **Balanced Dataset Handling**: Oversampling and scaling improve model accuracy on imbalanced data.
    
*   **Error Handling**: Mock response fallback ensures functionality if the Hugging Face API fails.
    

📦 Prerequisites
----------------

*   Python 3.8+
    
*   pip (Python package manager)
    
*   Git (for cloning the repository)
    
*   Hugging Face account for API key (free at [huggingface.co](https://huggingface.co/))
    

⚙️ Installation
---------------

### 1\. Clone the Repository

`git clone https://github.com/A-RYAN-1/Lung-Cancer-Prediction-Using-Machine-Learning.git  cd Lung-Cancer-Prediction-Using-Machine-Learning`

### 2\. Create and Activate a Virtual Environment

`python -m venv venv`

On Windows:

`venv\Scripts\activate`

On macOS/Linux:

`source venv/bin/activate`

### 3\. Install Dependencies

`pip install -r requirements.txt`

The requirements.txt includes:

`flask==3.0.3  numpy==1.26.4  pandas==2.2.2  scikit-learn==1.6.1  joblib==1.4.2  imbalanced-learn==0.12.3  requests==2.32.3  python-dotenv==1.0.1`

### 4\. Set Up Hugging Face API Key

1.  Sign up at [huggingface.co](https://huggingface.co/) and generate an API token (Settings > Access Tokens > New Token, select "Read" or "Write").
    
2.  HUGGINGFACE\_API\_KEY=hf\_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    

**🚫 Security Note: Do Not Push .env to GitHub**

*   The .env file contains sensitive API keys that could be exploited if exposed.
    
*   GitHub’s secret scanning may block pushes containing API keys or alert you.
    
*   Use .gitignore to exclude .env (see below).
    
*   echo "HUGGINGFACE\_API\_KEY=your\_huggingface\_token\_here" > .env.examplegit add .env.examplegit commit -m "Added .env.example for API key structure"
    

**Prevent .env Upload**:

`echo ".env" >> .gitignore  echo "*.pkl" >> .gitignore  echo "__pycache__/" >> .gitignore  echo "venv/" >> .gitignore  git add .gitignore  git commit -m "Updated .gitignore to exclude sensitive files"`

**Clean .env from Git History (if accidentally committed)**:

`pip install git-filter-repo  C:\Users\amazi\AppData\Roaming\Python\Python312\Scripts\git-filter-repo.exe --path .env --invert-paths --force  git push origin main --force`

**Note**: Use --force cautiously and back up your repository before rewriting history.

### 5\. Prepare the Dataset

*   Place cancer\_data.csv in the project root with columns: Age, Gender, Air Pollution, ..., Snoring, and Level (Low, Medium, High).
    

### 6\. Train the Model (if not using pre-trained models)
`python train_and_save_model.py`

This generates:

*   models/lung\_cancer\_model.pkl
    
*   models/scaler.pkl

    
▶️ Usage
--------

1.  python app.pyRuns the server at http://127.0.0.1:5000 in debug mode.
    
2.  **Access the Web Interface**:
    
    *   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000/) in a browser.
        
    *   Fill the form:
        
        *   **Age**: Numeric input (e.g., 50).
            
        *   **Gender**: Select Male (1) or Female (2).
            
        *   Other fields (e.g., Smoking, Air Pollution): Select "Yes" (1) or "No" (0).
            
    *   Click "Predict".
        
3.  **View Results**:
    
    *   **Prediction**: Shown in a blue box (e.g., "Predicted Lung Cancer Risk: Low Risk").
        
    *   **AI Explanation**: Shown in a green box (e.g., "Your low risk is likely due to minimal risk factors. Consider maintaining a healthy lifestyle and consulting a healthcare provider.").
        

🧠 How It Works
---------------

*   **Frontend**: templates/index.html provides a form styled with Tailwind CSS for a responsive UI.
    
*   **Backend**: app.py loads the model and scaler, processes form inputs, scales data, predicts risk, and queries the Hugging Face API.
    
*   **AI Integration**: Hugging Face’s Mixtral-8x7B generates explanations, with a mock response fallback for API failures.
    
*   **Model Training**: train\_and\_save\_model.py preprocesses data (encoding, scaling, oversampling with RandomOverSampler) and saves the model.
    
🤝 Contributing
---------------

1.  Fork the repository.
    
2.  git checkout -b feature/your-feature
    
3.  git commit -m "Add your feature"
    
4.  git push origin feature/your-feature
    
5.  Open a pull request.
   

📷 Working
----------------------

https://github.com/user-attachments/assets/9cc18f2d-15c3-4a14-a15a-2cebb7f74424
