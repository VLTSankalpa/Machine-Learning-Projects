# Blood transfusion service

The concept called **Gini importance** of variables and features in random forest classifiers can used to calculate the variable importance and then select most importance variables among other large number of variables or features in the random forest problem.

To explain the process an example dataset related to blood transfusion service centre has used as follow **[Q3.csv].** To demonstrate the RFMTC marketing model (a modified version of RFM), this study adopted the donor database of blood transfusion service centre in Hsin-Chu city in Taiwan. The RFM ( **Recency, Frequency, Monetary** ) analysis is a marketing technique used to determine quantitatively which customers are the best ones by examining how recently a customer has purchased (recency), how often they purchase (frequency), and how much the customer spends (monetary). To build a FRMTC model, it has selected 748 donors from the donor database. These 748 donor data, each one included following variables,

1. **Recency** - months since last donation
2. **Frequency** - total number of donation
3. **Monetary** - total blood donated
4. **Time** - months since first donation
5. **Binary variable** - represent whether he/she donated blood in March 2007 ( **1** stand for donating blood; **0** stands for not donating blood)

A random forest classifier has used to build the RFMTC marketing model by considering above one to four variables as **independent variables** and fifth variable as **dependent variable** as in **Q3.py** file.
