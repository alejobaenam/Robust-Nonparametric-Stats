# Robust and Nonparametric Statistics in Data Science

This repository contains code implementations for the master's course **Robust and Nonparametric Statistics**.

## 📊 About the Course
The course focuses on statistical methods that do not rely on traditional parametric assumptions. It covers robust techniques that are resilient to outliers and nonparametric methods useful when the underlying data distribution is unknown.

### **Key Topics Covered:**
- **Robust and Nonparametric Descriptive Statistics**
- **Outlier Detection (Univariate & Multivariate)**
- **Nonparametric Correlations (Kendall, Spearman, MAD, COMEDIAN)**
- **Bootstrap & Jackknife for Confidence Intervals**
- **Robust and Nonparametric Regression Techniques**
- **Spline Regression and Density Estimation with Kernels**
- **Rank-Based Statistical Tests (Wilcoxon, Mann-Whitney)**
- **Depth-Based Statistical Functions and Trimming Methods**

---

## 📁 Repository Structure

```
├── AllFunctions.py       # Python functions for robust and nonparametric methods
├── Exam1.ipynb           # Code related to Module 1: Introduction to Robust & NP Techniques
├── Exam2.ipynb           # Code related to Module 2: Robust Regression & Density Estimation
├── Exam3.ipynb           # Code related to Module 3: Rank-Based Statistical Tests
├── Spline.ipynb          # Implementation of Spline Regression and Smoothing Techniques
└── FullClasses.ipynb     # Comprehensive notebook with code from all classes
```

### **Files Description:**
- **`AllFunctions.py`**: Centralized Python functions for data analysis, estimation, and statistical tests.
- **`Exam1.ipynb`**: Covers nonparametric estimators, outlier detection, and bootstrap methods.
- **`Exam2.ipynb`**: Focuses on robust regression techniques, splines, and density estimation.
- **`Exam3.ipynb`**: Implements rank-based tests and depth-based statistical methods.
- **`Spline.ipynb`**: Dedicated notebook for spline models, smoothing techniques, and variance estimation.
- **`FullClasses.ipynb`**: A comprehensive notebook containing code implementations and examples from all class sessions.

---

## 🚀 Getting Started

### **Prerequisites:**
Ensure you have Python 3.x installed. Install the required libraries:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels scikit-learn
```

### **Cloning the Repository:**
```bash
git clone https://github.com/alejobaenam/Robust-Nonparametric-Stats.git
cd Robust-Nonparametric-Stats
```

### **Running the Notebooks:**
You can open the notebooks using Jupyter:

```bash
jupyter notebook
```

Or if you prefer VS Code or JupyterLab, open them directly from there.

---

## 📈 Example Usage
Here’s a simple example of using a robust estimator from `AllFunctions.py`:

```python
from AllFunctions import mad
import numpy as np

# Sample data with outliers
data = np.array([10, 12, 11, 9, 14, 100, 8, 13])

# Apply robust mean estimation
result = mad(data)
print(f"Robust Mean: {result}")
```

---

## 📚 References
- Huber, P. J. (1981). *Robust Statistics*. Wiley.
- Wilcox, R. R. (2013). *Introduction to Robust Estimation and Hypothesis Testing*. Academic Press.
- Wasserman, L. (2006). *All of Nonparametric Statistics*. Springer.