Project Overview
This project involves data cleaning, analysis, and visualization of two datasets: application_data and previous_application. The primary goal is to prepare the data for further analysis by handling missing values, removing irrelevant columns, and performing exploratory data analysis (EDA). The project also involves merging the datasets to analyze the relationships between features and understand the factors associated with customer payment difficulties.

Data Cleaning and Preparation
Missing Values Handling:

Columns with more than 50% missing data were dropped.
For remaining columns with missing values, appropriate imputation techniques were used (e.g., median or mean imputation).
Data Type Verification:

Ensured all columns had appropriate data types and corrected any discrepancies.
Dropping Irrelevant Columns:

Removed columns that were not relevant to the analysis or had too many null values.
Transforming Columns:

Converted negative values in certain columns to positive using the absolute function.
Categorized continuous variables into bins for better analysis (e.g., AMT_INCOME_TOTAL and AMT_CREDIT).
Data Analysis
Target Variable Distribution:

Analyzed the distribution of the target variable to understand the proportion of customers with payment difficulties.
Exploratory Data Analysis (EDA):

Visualized the distribution of various features across different categories such as income type, contract type, and family status.
Analyzed the correlation between features, particularly focusing on the top 10 correlated variables.
Gender Distribution:

Examined the distribution of gender in both the application_data and combined datasets.
Income Type Distribution:

Analyzed how income types are distributed among customers and their relationship with repayment status.
Correlation Analysis:

Identified and visualized the top correlated variables using a heatmap to understand the relationships between different features.
Visualization
Various plots were created to visualize the distribution of data, including histograms, pie charts, and heatmaps.
A custom plotting function was developed to facilitate the visualization of categorical variables across different segments of the data.
Key Findings
Identified strong correlations between features such as AMT_GOODS_PRICE and AMT_APPLICATION, indicating that the requested loan amount is proportional to the price of the product.
Observed a higher proportion of female customers compared to male customers in the combined dataset.
Found that married customers constitute the majority in both repayment statuses, though there's a slight decrease among those with payment difficulties.
Conclusion
This project effectively cleansed and prepared the data for more detailed analysis. The EDA provided insights into the characteristics of customers with payment difficulties and the relationships between various features. The findings can guide further analysis, such as building predictive models or conducting more detailed statistical analysis.
