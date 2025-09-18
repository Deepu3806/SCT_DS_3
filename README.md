# 🌳 Decision Tree Classifier – Bank Marketing Dataset

This repository contains my **Task 03** from the **Data Science Internship at SkillCraft Technology**.  
The project applies a **Decision Tree Classifier** to the **Bank Marketing dataset** to predict whether a client will subscribe to a term deposit.

---

## 📌 Project Highlights
- Preprocessed categorical features using **Label Encoding**
- Trained a **Decision Tree Classifier** with entropy criterion
- Evaluated model using **Accuracy, Classification Report, and Confusion Matrix**
- Visualized:
  - **Feature Importances** (bar chart)
  - **Readable Decision Tree** (Graphviz)
- Extracted **human-readable rules** for interpretability
- Saved trained model as `bank_dt_model.joblib`

---

## 📂 Outputs
- `feature_importance_bargraph.png` → Bar chart of feature importance  
- `decision_tree_image.png` → Clean, user-friendly decision tree    
- `bank_dt_model.joblib` → Saved trained model  

---

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/decision-tree-bankmarketing.git
   cd decision-tree-bankmarketing
2. Install required dependencies:
   pip install -r requirements.txt
   (requirements: pandas, scikit-learn, matplotlib, seaborn, graphviz, joblib)
3.Run the script:
  python decision_tree.py

📊 Sample Results

Accuracy: ~85% (varies with dataset split)

Feature Importance: Highlights key features influencing prediction (e.g., duration, age, balance)

Readable Tree: Visual tree generated via Graphviz for easy interpretation

🎯 Learning Outcomes

Applied supervised ML (Decision Tree) to a real-world dataset

Improved skills in feature engineering, model training, evaluation, and visualization

Learned how to make ML models more interpretable for users

👩‍💻 Author

Deepika
Data Science Intern @ SkillCraft Technology
   
