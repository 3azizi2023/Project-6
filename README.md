# Project-6
Dashboard link created : https://btzmufxddbxjhlgu2zlxsv.streamlit.app/

At "Ready to Spend," a financial company specializing in consumer loans, our goal is to make these loans accessible to individuals with little or no credit history. The company "Ready to Spend" aims to establish a "credit scoring" tool based on various data sources to calculate the probability of customer repayment and classify credit applications as approved or denied. Simultaneously, the company is responding to the increasing demand for transparency from customers by developing an interactive dashboard. This dashboard will enable customer relationship managers to transparently explain credit approval decisions and provide customers with easy access to their personal information.

Mission:

Build a scoring model that automatically predicts a customer's likelihood of default.
Develop an interactive dashboard for customer relationship managers to interpret the predictions made by the model and enhance customer knowledge for relationship managers.
Deploy the predictive scoring model using an API, along with the interactive dashboard that calls the API for predictions.
Data Description:

application_{train|test}.csv: Main table divided into training (with the target variable "TARGET") and testing sets.
bureau.csv: Contains information on the client's previous credits from other financial institutions reported to the Credit Bureau.
bureau_balance.csv: Monthly balances of previous credits reported to the Credit Bureau.
POS_CASH_balance.csv: Monthly snapshots of balances for previous loans in points of sale (POS) and in cash held by the applicant at Home Credit.
credit_card_balance.csv: Monthly snapshots of balances for previous credit cards owned by the applicant at Home Credit.
previous_application.csv: Information on all previous loan applications made by clients with loans in our dataset.
installments_payments.csv: History of repayments for previously granted credits at Home Credit.
HomeCredit_columns_description.csv: Descriptions of the columns in the various data files.
