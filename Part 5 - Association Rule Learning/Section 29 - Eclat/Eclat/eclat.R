# Eclat

# Data Preprocessing
data = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
#install.packages('arules')
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2)) # Support calculated as - (3*7/7500) - (product purchased 3/4/5/etc times a day, over a week = 3 * 7 / number of transactions = 7500)

# Visualizing the results
inspect(sort(rules, by = 'support')[1:10])