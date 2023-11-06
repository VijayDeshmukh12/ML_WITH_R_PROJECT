install.packages("palmerpenguins")
# Load the Libraries
library(palmerpenguins)
data("penguins")
#data <- read.csv("penguins.csv")
head(penguins)
data <- penguins
#View the dataset
View(data)
str(data)
#Checking missing values
is.na(data)
#sum of missing values
sum(is.na(data))
# Delete all rows with missing data
data <- na.omit(data)
#Exploring categorical variables
table(penguins$species)
table(penguins$island)
table(penguins$sex)
#Exploring relationships between variables
#Exploratory Data Analysis (EDA)
# Distribution of numerical variables
par(mfrow=c(2,2))
hist(penguins$bill_length_mm, main="Bill Length (mm)", xlab="Bill Length (mm)")
hist(penguins$bill_depth_mm, main="Bill Depth (mm)", xlab="Bill Depth (mm)")
hist(penguins$flipper_length_mm, main="Flipper Length (mm)", xlab="Flipper Length (mm)")
hist(penguins$body_mass_g, main="Body Mass (g)", xlab="Body Mass (g)")

# Boxplots for categorical variables
par(mfrow=c(1,2))
boxplot(penguins$species ~ penguins$sex, main="Species by Sex", xlab="Sex", ylab="Species")
boxplot(penguins$island ~ penguins$sex, main="Island by Sex", xlab="Sex", ylab="Island")

# Scatterplot matrix for numerical variables
pairs(penguins[, c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")])

# Correlation matrix
cor(penguins[, c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")])

# Visualize the distribution of species
library(ggplot2)
ggplot(penguins, aes(x = species, fill = species)) + geom_bar() + labs(title="Distribution of Species")

# Visualize the distribution of islands
ggplot(penguins, aes(x = island, fill = island)) + geom_bar() + labs(title="Distribution of Islands")

# Visualize the relationship between numerical variables
ggplot(penguins, aes(x = bill_length_mm, y = bill_depth_mm, color = species)) + geom_point() + labs(title="Bill Length vs Bill Depth")

ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g, color = species)) + geom_point() + labs(title="Flipper Length vs Body Mass")
# Heatmap for correlations (if desired)
cor_matrix <- cor(penguins[, c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")])
# Check for missing values in the correlation matrix
if(sum(is.na(cor_matrix)) > 0) {
  # If there are missing values, replace them with zeros
  cor_matrix[is.na(cor_matrix)] <- 0
}
heatmap(cor_matrix, col = colorRampPalette(c("blue", "white", "red"))(50))

# Check for duplicate rows
duplicates <- sum(duplicated(data))
print(paste("Total duplicate rows:", duplicates))
# Remove duplicate rows
data <- data[!duplicated(data), ]
# Set seed for reproducibility
set.seed(123)
# Split data into training and testing sets
indexes <- sample(1:nrow(data), size = 0.7 * nrow(data))
train_data <- data[indexes, ]
test_data <- data[-indexes, ]
train_labels <- data[indexes, 1]
test_labels <- data[-indexes, 1]
# Normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# Load libraries
install.packages("class")
library(class)
k <- 5
# Apply kNN
predicted_labels <- knn(train_data[, c("bill_length_mm", "bill_depth_mm")], 
                         test_data[, c("bill_length_mm", "bill_depth_mm")], 
                         train_data$species, k)
# Load the 'gmodels' package for the CrossTable function
install.packages("gmodels")
library(gmodels)
#Create a cross table to analyze the results
cross_table <- CrossTable(test_data$species, predicted_labels, dnn=c('Actual', 'Predicted'))
#Use a confusion matrix to compare predicted labels with actual labels
conf_matrix <- table(test_data$species, predicted_labels)
print(conf_matrix)
#Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(accuracy)
# Load necessary packages
install.packages("tm")
library(tm)
install.packages("wordcloud")
library(wordcloud)
# Create a corpus from the 'species', 'island', and 'sex' columns
corpus <- Corpus(VectorSource(paste(penguins$species, penguins$island, penguins$sex)))
# Create a term-document matrix
tdm <- TermDocumentMatrix(corpus)
# Convert to a matrix and calculate word frequencies
m <- as.matrix(tdm)
word_freqs <- sort(rowSums(m), decreasing=TRUE)
# Generate the wordcloud
wordcloud(words = names(word_freqs), freq = word_freqs, min.freq = 1, scale = c(3,0.5), colors=brewer.pal(8, "Dark2"))
#Build and visualize a decision tree
install.packages("rpart")
library(rpart)
install.packages("ggplot2")
library(ggplot2)
install.packages("rpart.plot")
library(rpart.plot)
# Create a decision tree model
tree_model <- rpart(species ~ ., data = penguins, method = "class")
# Visualize the decision tree
rpart.plot(tree_model)
# Fit a linear regression model
reg_model <- lm(body_mass_g ~ flipper_length_mm, data = penguins)
# Visualize the linear regression line
ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) +
  geom_point() +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  labs(title = "Regression of Body Mass vs. Flipper Length",
       x = "Flipper Length (mm)",
       y = "Body Mass (g)")   
# Load necessary packages
install.packages(c("factoextra", "cluster"))
library(factoextra)
library(cluster)
# Preprocess the data (remove any missing values)
penguins_clean <- na.omit(penguins)
# Perform PCA for dimensionality reduction
pca_result <- prcomp(penguins_clean[, c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")])
# Use the first two principal components for clustering
pca_data <- data.frame(PC1 = pca_result$x[, 1], PC2 = pca_result$x[, 2])
# Perform k-means clustering
kmeans_clusters <- kmeans(pca_data, centers = 3, nstart = 25)
# Visualize the clusters
fviz_cluster(kmeans_clusters, data = pca_data, geom = "point")
# Load necessary packages
install.packages("neuralnet")
library(neuralnet)
# Create a neural network model
nn_model <- neuralnet(body_mass_g ~ flipper_length_mm + bill_depth_mm, data = penguins_clean, hidden = c(5, 3))
# Visualize the neural network
plot(nn_model, rep = "best")
#Hierarchical Clustering Dendrogram
# Calculate the Euclidean distance matrix
d = dist(penguins_clean, method = "euclidean")
# Perform hierarchical clustering using average linkage
hfit = hclust(d, method = "average")
# Plot the dendrogram
plot(hfit)
# Cut the dendrogram into 4 clusters
grps = cutree(hfit, k = 4)
# Display the cluster assignments
grps
# Add rectangles around the clusters in the dendrogram
rect.hclust(hfit, k = 4, border = "red")