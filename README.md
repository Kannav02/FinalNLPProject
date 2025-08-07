# NLP Final Project: AI vs Human Text Classification

This project implements and compares different machine learning models for classifying text as AI-generated or human-written. The project includes baseline models and a fine-tuned transformer model with comprehensive evaluation and analysis.

This is my final year NLP project submission, so open-sourced, but I think no contributions would be required, thanks!


### Prerequisites
I am using uv as the package manager and runtime manager as well, so the instruction to install are also provided in it

**NOTE** uv is pretty good and fast, I enjoyed using it in this project

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv**:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup the project**:
   ```bash
   git clone <repository-url>
   cd nlp-final-project
   uv sync
   ```

### Running the Project

1. **Activate the virtual environment**:
   ```bash
   uv shell
   ```

2. **Run Jupyter notebooks**:
   ```bash
   jupyter notebook
   ```
   
   Then open any of the notebooks in the `notebooks/` directory.

3. **Run evaluation scripts**:
   ```bash
   python evaluation_metrics.py
   python misclassification_analysis.py
   ```

## Project Structure

I will not include the dataset or the model, as they are pretty heavy files and git wouldn't allow me to upload them.

```
nlp-final-project/
├── data/                           # Dataset directory
│   └── AI_Human.csv               # Main dataset with AI and human text samples
├── notebooks/                      # Jupyter notebooks for model development
│   ├── baseline_model.ipynb       # Local development of baseline models
│   ├── baseline_model_final.ipynb # Final baseline models (cloud execution)
│   ├── fine_tuned_model.ipynb     # Local development of fine-tuned model
│   └── fine_tuned_model_final.ipynb # Final fine-tuned model (cloud execution)
├── models/                         # Trained model artifacts
│   └── checkpoint-36543/          # Fine-tuned DistilBERT model checkpoint
├── reports/                        # Project documentation and reports
│   ├── comprehensive_final_report.md # Complete project report
│   ├── final_report.pdf           # PDF version of final report
│   ├── model_comparison.md        # Comparison of all models
│   ├── model_performance_comparison.md # Baseline models comparison
│   └── project_plan.md           # Original project planning document
├── evaluation_metrics.py          # Modular evaluation functions
├── misclassification_analysis.py  # Edge case analysis for fine-tuned model
├── model_comparison.md            # Model comparison summary
├── pyproject.toml                 # Project dependencies and configuration
└── uv.lock                       # Dependency lock file
```

## Models Implemented

1. **Baseline Models**:
   - LinearSVC
   - Logistic Regression

2. **Advanced Model**:
   - Fine-tuned DistilBERT for sequence classification

---

For detailed methodology, results, and analysis, please refer to the reports in the `reports/` directory.