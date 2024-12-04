# **Laporan Proyek Machine Learning - Kevin Aditya Ikhsan**

## **Domain Project**  

Marketing campaigns play a crucial role in driving business growth, particularly in the financial industry. In this context, telemarketing, which involves contacting customers remotely through channels such as telephone calls, has been a widely adopted approach. However, outbound telemarketing often faces challenges, such as high costs and intrusiveness, which can negatively impact customer relationships [[1]](https://repositorium.sdum.uminho.pt/bitstream/1822/30994/1/dss-v3.pdf). Each call incurs an estimated cost of €5 [[2]](https://www.maestroqa.com/blog/call-center-cost-per-call), making it essential to optimize targeting strategies to ensure that only customers with the highest likelihood of success are contacted. Furthermore, considering a minimum deposit requirement of €5,000 [[3]](https://www.bancomontepio.pt/en/individuals/savings-and-retirement/term-deposits) and an average Net Interest Margin (NIM) of 1.25% in Portugal [[4]](https://www.theglobaleconomy.com/Portugal/net_interest_margin/), successful calls represent significant revenue opportunities, generating an estimated net revenue of €62.50 per deposit annually. 

This project proposes a machine learning-driven solution to enhance the effectiveness of telemarketing campaigns by predicting the likelihood of a successful long-term deposit subscription. By leveraging machine learning models, the system aims to analyze customer data and prioritize individuals with higher conversion probabilities, thereby reducing campaign costs and improving efficiency. This approach enables banks to focus resources on high-potential leads, minimizing unnecessary calls while maximizing revenue opportunities and maintaining better customer relationships.

## **Business Understanding**

### **Problem Statements**

1. **High Costs and Inefficiency:** Outbound telemarketing incurs significant costs (€5 per call) and inefficiency due to a lack of accurate targeting, leading to wasted resources and potential customer dissatisfaction.  
2. **Maximizing Revenue Opportunities:** The need to identify high-potential customers to maximize revenue from long-term deposits (€62.50 per successful deposit annually) remains unaddressed.  

### **Goals**

1. **Optimize Telemarketing Costs:** Reduce the number of unnecessary calls by accurately predicting the likelihood of success for each customer.  
2. **Increase Revenue Efficiency:** Improve the overall return on investment by prioritizing calls to high-potential customers, ensuring higher conversion rates.

### **Solution Statements**

1. **Machine Learning Model Development:** Implement and compare machine learning algorithms, such as Logistic Regression, KNN, Naive Bayes, Decision Tree, and Ensemble Methods, to predict customer conversion probabilities.  
2. **Hyperparameter Tuning:** Enhance the performance of the chosen baseline model through systematic hyperparameter optimization.  
3. **Evaluation Metrics:** Use **recall** to ensure high identification of potential customers and calculate business metrics (e.g., cost savings and revenue gains) to assess the economic impact of the solution.
4. **Feature Importance Analysis:** Perform feature importance analysis to identify the most impactful factors influencing customer conversions, enabling more informed decision-making.

## **Data Understanding**

This project utilizes the **Bank Marketing Campaigns Dataset** [[5]](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset), which captures the results of marketing campaigns conducted by a bank in Portugal. These campaigns primarily involved direct phone calls to clients, promoting term deposit subscriptions. The target, **`y`**, indicates whether a client subscribed to a term deposit ("yes" or "no").

### **Dataset Features**  
The dataset contains 21 features categorized into different types based on their relevance:

1. **Bank Client Data**:
   - **`age`**: Age of the client (numeric).
   - **`job`**: Type of job (categorical: e.g., "admin.", "blue-collar", "entrepreneur", "unknown").
   - **`marital`**: Marital status (categorical: e.g., "single", "married", "divorced", "unknown").
   - **`education`**: Education level (categorical: e.g., "university.degree", "high.school", "unknown").
   - **`default`**: Credit default status (categorical: "yes", "no", "unknown").
   - **`housing`**: Housing loan status (categorical: "yes", "no", "unknown").
   - **`loan`**: Personal loan status (categorical: "yes", "no", "unknown").

2. **Last Contact Data**:
   - **`contact`**: Communication type (categorical: "cellular", "telephone").
   - **`month`**: Last contact month (categorical: e.g., "jan", "feb", "dec").
   - **`day_of_week`**: Last contact day (categorical: "mon", "tue", "fri").
   - **`duration`**: Last contact duration in seconds (numeric). (Note: For realistic predictive models, this feature is excluded as it’s only known after the call.)

3. **Campaign Data**:
   - **`campaign`**: Number of contacts made during the campaign (numeric).
   - **`pdays`**: Days since last client contact in a previous campaign (numeric; 999 indicates no prior contact).
   - **`previous`**: Number of contacts before the current campaign (numeric).
   - **`poutcome`**: Outcome of the previous campaign (categorical: "success", "failure", "nonexistent").

4. **Social and Economic Context**:
   - **`emp.var.rate`**: Employment variation rate (numeric).
   - **`cons.price.idx`**: Consumer price index (numeric).
   - **`cons.conf.idx`**: Consumer confidence index (numeric).
   - **`euribor3m`**: Euribor 3-month rate (numeric).
   - **`nr.employed`**: Number of employees (numeric).

5. **Target**:
   - **`y`**: Whether the client subscribed to a term deposit (binary: "yes", "no").

### **Handling Missing Values**  
Missing values in categorical features, denoted as "unknown," will be addressed through appropriate imputation techniques or treated as a separate class if relevant to the analysis.
