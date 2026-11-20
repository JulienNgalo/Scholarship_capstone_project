# AI Assistance Disclosure

## 1. AI Tools Used

During the development of my Machine Learning Scholarship Shortlisting Capstone Project, I used the following AI tools:

- **OpenAI ChatGPT**
- **Copilot SDK integrated into Visual Studio Code**

These tools supported my learning, software development, troubleshooting, technical documentation, and presentation preparation.

## 2. My Personal Contributions

### A. Problem Identification and Project Objectives

I identified the scholarship shortlisting challenges experienced by the Financial Aid Office at Strathmore University, particularly the reliance on manual Excel-based screening.

I defined the project's problem statement, research objectives, scope, and intended contribution: developing a machine learning-based decision-support system to improve the consistency, efficiency, and explainability of scholarship applicant screening.

### B. Data Collection and Preparation

I provided the historical scholarship application records and relevant reference datasets, including secondary school classifications and county-level information.

I worked on exploring the datasets, identifying data quality issues, handling duplicate records, and reviewing missing values and inconsistent classifications.

I also made decisions regarding the features used in the prediction model, including academic performance, O-Level completion gaps, county information, parental status, and secondary school categories.

### C. Machine Learning Development and Evaluation

I worked on developing and evaluating four classification algorithms:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Gradient Boosting

I reviewed their performance using accuracy, precision, recall, F1-score, and ROC-AUC.

I selected Gradient Boosting for the final implementation after reviewing the modelling results and considering the project's objectives.

I also worked on improving the evaluation methodology through student-group-aware data splitting to reduce the risk of data leakage.

### D. Model Explainability, Fairness, and Monitoring

I examined the model's predictions using SHAP explanations to understand how individual features contributed to predicted scholarship outcomes.

I conducted subgroup analyses involving county, parental status, and secondary school classification to assess differences in model performance.

I also explored data drift monitoring to identify differences in feature distributions and consider how model performance could be monitored over time.

I reviewed and interpreted these analyses in the context of scholarship screening and responsible decision-making.

### E. Application Development and Deployment

I worked on implementing and testing a FastAPI prediction service and a Streamlit dashboard for individual and batch scholarship applicant screening.

I configured and tested the Docker-based application locally, reviewed model outputs, and checked the functionality of the API and dashboard.

### F. Automated Testing and Continuous Integration

I executed automated Python tests locally and implemented GitHub Actions as part of the project's Continuous Integration workflow.

I verified that all six automated tests passed both in my local development environment and on GitHub Actions.

I also managed the GitHub repository, reviewed commits, and maintained the project's version-control workflow.

### G. Documentation and Presentation

I reviewed and refined the project documentation, including the README, setup instructions, architecture description, reproducibility procedures, and final PowerPoint presentation.

I remained responsible for ensuring that the submitted materials reflected the methods, findings, implementation, and limitations of my project.

## 3. How AI Assisted My Work

ChatGPT and Copilot SDK provided assistance in the following areas:

- Explaining machine learning concepts, modelling techniques, validation approaches, and evaluation metrics.
- Suggesting, generating, refining, and troubleshooting Python code used in data preparation, feature engineering, model training, and evaluation.
- Assisting with implementation and debugging of the FastAPI service, Streamlit dashboard, SHAP explanations, and Docker configuration.
- Providing guidance on subgroup fairness analysis, data drift monitoring, and interpretation of model results.
- Helping develop and refine automated tests and GitHub Actions workflows.
- Explaining terminal commands, Git operations, dependency issues, and software execution errors.
- Assisting with drafting, structuring, reviewing, and improving technical documentation and presentation materials.

AI assistance was used throughout the development process, but I participated in reviewing the suggested solutions, executing commands, testing functionality, evaluating outputs, and making project decisions.

## 4. Responsibility and Verification

The original applicant data, secondary school reference information, historical scholarship outcome labels, and institutional problem context were supplied by me and were not independently verified by the AI tools.

I take responsibility for the project's research objectives, data handling, feature selection, methodological decisions, model evaluation, interpretation of findings, and final submission.

I acknowledge that AI-generated code, explanations, and written content may contain errors. I therefore reviewed and tested the outputs used in the project and remain responsible for identifying and correcting any remaining inaccuracies.

## 5. Ethical Considerations

The scholarship prediction system was developed as an academic research prototype intended to support human decision-making.

I recognise that historical scholarship outcomes may reflect existing institutional practices and biases. Model predictions should therefore not be treated as automatic decisions to approve or reject scholarship applications.

Any future operational use would require appropriate institutional authorisation, data-protection safeguards, fairness evaluation, and human oversight.

## 6. Final Declaration

I declare that this capstone project reflects my participation in identifying the research problem, working with the data, developing and evaluating machine learning models, testing the application, interpreting the results, and preparing the final submission.

I have disclosed the use of ChatGPT and Copilot SDK in activities where AI contributed explanations, code suggestions, troubleshooting support, and written material.

I remain accountable for the accuracy, integrity, originality, and conclusions of the work submitted.

This disclosure is provided in the interest of academic transparency and is subject to the university's applicable academic integrity and AI-use requirements.