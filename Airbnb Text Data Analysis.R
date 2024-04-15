#installing and loading the mongolite library to download the Airbnb data
#install.packages("mongolite") #need to run this line of code only once and then you can comment out
library(mongolite)

# This is the connection_string. You can get the exact url from your MongoDB cluster screen
#replace the <<user>> with your Mongo user name and <<password>> with the mongo password
#lastly, replace the <<server_name>> with your MongoDB server name

connection_string <- 'mongodb+srv://Rachana26:18242628@cluster0.vmeuvvh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
airbnb_collection <- mongo(collection="listingsAndReviews", db="sample_airbnb", url=connection_string)

#Here's how you can download all the Airbnb data from Mongo
## keep in mind that this is huge and you need a ton of RAM memory

airbnb_all <- airbnb_collection$find()

#-----------------------------------------------------------------------------------------#
# Load the necessary  Libraries 
#-----------------------------------------------------------------------------------------#
library(tidytext)
library(dplyr)
library(ggplot2)
library(ggplot2)
library(topicmodels)
library(tibble)
library(tidyr)
library(tidyverse)
library(reshape2)
library(jsonlite)
library(wordcloud)
library(RColorBrewer)

#---------------------------------------------------------#
#------------------Data exploration ----------------------#
#---------------------------------------------------------#

# Inspect the first few rows of the dataset
head(airbnb_all)

colnames(airbnb_all)
#is.numeric(airbnb_all$description)

# Count total missing values in the entire data frame
sum(is.na(airbnb_all))
colSums(is.na(airbnb_all))
# Fill missing values in numerical columns with the median of the column
num_cols <- sapply(airbnb_all, is.numeric) # Identify numerical columns
num_cols
airbnb_all[num_cols] <- lapply(airbnb_all[num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
# Check again for missing values in each column
colSums(is.na(airbnb_all))
sum(is.na(airbnb_all))

#----------------------------------------------------------------#
#---------------Texts data analysis -----------------------------#
#----------------------------------------------------------------#
# Tokenize the descriptions
tidy_data <- airbnb_all %>%
  unnest_tokens(word, description)

# Removing stop words
data(stop_words)
tidy_data <- tidy_data %>%
  anti_join(stop_words, by = "word") 

#--------------------------------------------------#
#-------Calculating frquency of words -------------#
#--------------------------------------------------#
# Count the frequency of each word
word_frequencies <- tidy_data %>%
  group_by(word) %>%
  summarise(frequency = n()) %>%
  ungroup() %>%
  arrange(desc(frequency))

# View the top 20 most frequent words
top_20_words <- head(word_frequencies, 20)
print(top_20_words)

# Plot the top 20 most frequent words
ggplot(top_20_words, aes(x = reorder(word, frequency), y = frequency)) +
  geom_col(fill = "skyblue") +
  coord_flip() +  # Flip the axes to make the words readable
  labs(title = "Top 20 Most Frequent Words in Airbnb Listings",
       x = "Word",
       y = "Frequency") +
  theme_minimal()


#----------------------------------------------------#
#-------Converting data to DTM-----------------------#
#----------------------------------------------------#

# At this point, you can proceed to create the DTM for further analysis
dtm <- tidy_data %>%
  count(document = listing_url, word) %>%
  cast_dtm(document, word, n)


tf_idf <- tidy_data %>%
  count(word, document = listing_url) %>% # Replace 'listing_id' with the appropriate identifier column for each listing
  bind_tf_idf(word, document, n) %>%
  arrange(desc(tf_idf))
# To see the top 10 words with the highest TF-IDF scores
top_tf_idf <- tf_idf %>%
  top_n(10, tf_idf)

# Optionally, visualize the results
#library(ggplot2)
ggplot(top_tf_idf, aes(x = reorder(word, tf_idf), y = tf_idf)) +
  geom_col() +
  coord_flip() +
  labs(x = "Term", y = "TF-IDF Score", title = "Top Terms by TF-IDF Score")

#----------------------------#
#------------ N gram---------#
#----------------------------#
four_grams <- airbnb_all %>%
  unnest_tokens(output = ngram, input = description, token = "ngrams", n = 4)

# Count the frequency of each 4-gram
four_grams_counts <- four_grams %>%
  count(ngram, sort = TRUE)
# Top 20 4-grams
top_20_four_grams <- head(four_grams_counts, 20)

# Visualization
#library(ggplot2)
ggplot(top_20_four_grams, aes(x = reorder(ngram, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 4-grams in Airbnb Listings Descriptions", x = "4-gram", y = "Frequency") +
  theme_minimal()

#------------------------------------------------------------#
#---------LDA------------------------------------------------#
#------------------------------------------------------------#
# Perform LDA on the DTM
# Here, k is the number of topics you want to model;
# For example, finding 5 topics
lda_model <- LDA(dtm, k = 5, control = list(seed = 1234))

# Examine the terms associated with each topic
terms(lda_model, 10)  # Retrieves 10 terms for each topic

# To see the topics associated with each document
topics <- tidy(lda_model, matrix = "gamma")
head(topics)

# Optionally, you can examine how topics are distributed across documents
# You may want to transform 'gamma' values to be 'tidy'
tidy_topics <- tidy(lda_model, matrix = "gamma") %>%
  group_by(document) %>%
  top_n(1, gamma) %>%
  ungroup()

# Then, you can visualize the results
#library(ggplot2)
ggplot(tidy_topics, aes(factor(topic), fill = factor(topic))) +
  geom_bar() +
  labs(x = "Topic", y = "Count", fill = "Topic") +
  theme_minimal()

# Get the terms from the LDA model
topics <- tidy(lda_model, matrix = "beta")

# Get the top terms for each topic
top_terms_per_topic <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Examine the terms to label the topics
top_terms_per_topic
#library(ggplot2)

# Example visualization for 5 topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)
#library(ggplot2)
# Visualize the terms
ggplot(top_terms, aes(x = reorder(term, beta), y = beta, fill = factor(topic))) +
  geom_bar(stat = "identity") +
  coord_flip() +  # To flip the axes
  facet_wrap(~ topic, scales = "free") +  # Create a separate plot for each topic
  labs(x = "Term", y = "Probability", title = "Top Terms in Each Topic from LDA Model") +
  theme_minimal() +
  guides(fill=guide_legend(title="Topic"))  # Add a legend for topics

#---------------------------------------------------#
#---------Sentiment analysis -----------------------#
#---------------------------------------------------#
# Get the Bing sentiment lexicon
bing <- get_sentiments("bing")

# and that 'count' is created to represent the word count
tidy_data <- tidy_data %>%
  group_by(listing_url, word) %>%
  summarize(word_count = n(), .groups = 'drop')  # Avoid using the name 'count'

# Now join your data with the Bing sentiment lexicon
sentiment <- tidy_data %>%
  inner_join(bing, by = "word")

bing_sentiment_score <- sentiment %>%
  group_by(listing_url, sentiment) %>%
  summarize(total_count = sum(word_count, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(names_from = sentiment, values_from = total_count, values_fill = list(total_count = 0)) %>%
  mutate(sentiment_score = positive - negative)


print(head(bing_sentiment_score))

# Factorize the room type
airbnb_all$room_type <- factor(airbnb_all$room_type)
# Merge sentiment scores with metadata from the main dataset
# Ensure that the 'listing_url' is the common key
sentiment_metadata <- left_join(bing_sentiment_score, airbnb_all, by = "listing_url")

# Boxplot of sentiment score by room type
ggplot(sentiment_metadata, aes(x = room_type, y = sentiment_score)) +
  geom_boxplot() +
  labs(title = "Sentiment Score by Room Type", x = "Room Type", y = "Sentiment Score")

#-----------------------------------------------------------------------#
#------------------- NRC sentiment  analysis ---------------------------#
#-----------------------------------------------------------------------#
# Get the NRC sentiment lexicon
nrc <- get_sentiments("nrc")

# Join your data with the NRC sentiment lexicon
nrc_sentiment <- tidy_data %>%
  inner_join(nrc, by = "word")

# Calculate sentiment score for each document
nrc_sentiment_score <- nrc_sentiment %>%
  group_by(listing_url, sentiment) %>%
  summarize(emotion_count = sum(word_count, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(names_from = sentiment, values_from = emotion_count, values_fill = list(emotion_count = 0))

# Explore NRC sentiment scores
head(nrc_sentiment_score)

# Gather the data into a long format
long_format <- nrc_sentiment_score %>%
  gather(key = "sentiment_emotion", value = "count", -listing_url)

# Verify the data structure
print(head(long_format))

# Filter for a single listing if desired, or plot for all listings
# Here, I'm showing how to plot for all listings for simplicity
ggplot(long_format, aes(x = sentiment_emotion, y = count)) +
  geom_bar(stat = "identity", fill = "turquoise3") +
  theme_minimal() +
  labs(title = "Sentiment and Emotion Counts across Listings",
       x = "Sentiment/Emotion",
       y = "Count") +
  coord_flip() # Flip the coordinates for horizontal bars


# Check if 'host' is a list-column
if("host" %in% names(airbnb_all) && is.list(airbnb_all$host)) {
  # If it is, unnest the 'host' column
  airbnb_all <- airbnb_all %>%
    tidyr::unnest_wider(host)
}

# Now attempt to select the columns again
host_data <- airbnb_all %>%
  select(listing_url, 
         host_is_superhost, 
         host_listings_count, 
         host_total_listings_count, 
         host_response_rate,
         host_identity_verified,
         host_has_profile_pic,
         host_response_time)


# merge the host_data with sentiment_scores
full_data <- merge(host_data, nrc_sentiment_score, by = "listing_url", all.x = TRUE)
print(head(full_data))


# missing values for 'host_response_rate', you can decide to fill them with a placeholder or median, for example
full_data$host_response_rate[is.na(full_data$host_response_rate)] <- median(full_data$host_response_rate, na.rm = TRUE)

# Convert logical columns to numeric
full_data$host_is_superhost <- as.integer(full_data$host_is_superhost)
full_data$host_identity_verified <- as.integer(full_data$host_identity_verified)
full_data$host_has_profile_pic <- as.integer(full_data$host_has_profile_pic)

# Convert character columns to factor and then to numeric if they have an order
# For example, if host_response_time has an order you can map it like below:
response_time_levels <- c("within an hour", "within a few hours", "within a day", "a few days or more")
full_data$host_response_time <- factor(full_data$host_response_time, levels = response_time_levels)
full_data$host_response_time <- as.integer(full_data$host_response_time)

# If host_response_time does not have an order, it could be more appropriate to drop this column for the correlation analysis
full_data$host_response_time <- NULL

# Fill NA values for host_response_rate with median
full_data$host_response_rate[is.na(full_data$host_response_rate)] <- median(full_data$host_response_rate, na.rm = TRUE)

# Now attempt the correlation analysis again, excluding non-numeric columns
numeric_columns <- sapply(full_data, is.numeric)
correlation_results <- cor(full_data[, numeric_columns], use = "complete.obs")

# View the correlation results
print(correlation_results)
# Melt the correlation matrix into a long format for ggplot
melted_correlation <- melt(correlation_results, na.rm = TRUE)

# Generate the heatmap with labels
ggplot(melted_correlation, aes(Var1, Var2, fill = value)) +
  geom_tile() + # This adds the tiles to the plot
  geom_text(aes(label = sprintf("%.2f", value)), vjust = 1, color = "black", size = 3) + # Add the correlation values as text on the tiles
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        axis.text.y = element_text(vjust = 0.5)) +
  labs(title = "Correlation Matrix Heatmap", x = "", y = "", fill = "Correlation") +
  coord_fixed()
#------------------------------------------------------------------#    
#----------- AFINN Sentiment Lexicon-------------------------------#
#------------------------------------------------------------------#
# Get the AFINN sentiment lexicon
afinn <- get_sentiments("afinn")

# Join your data with the AFINN sentiment lexicon
afinn_sentiment <- tidy_data %>%
  inner_join(afinn, by = "word")

# Calculate sentiment score for each listing
afinn_sentiment_score <- afinn_sentiment %>%
  group_by(listing_url) %>%
  summarize(overall_sentiment = sum(value * word_count, na.rm = TRUE), .groups = 'drop')
# Explore the overall sentiment scores
head(afinn_sentiment_score)


# Histogram of AFINN sentiment scores
ggplot(afinn_sentiment_score, aes(x = overall_sentiment)) +
  geom_histogram(bins = 30, fill = "cornflowerblue") +
  labs(title = "Distribution of AFINN Sentiment Scores for Airbnb Listings",
       x = "AFINN Sentiment Score",
       y = "Count") +
  theme_minimal()
# Assuming 'airbnb_all' is your dataset containing cancellation policies
cancellation_sentiment <- airbnb_all %>%
  select(listing_url, cancellation_policy) %>%
  inner_join(afinn_sentiment_score, by = "listing_url")

# Group by cancellation policy and calculate average sentiment
policy_sentiment <- cancellation_sentiment %>%
  group_by(cancellation_policy) %>%
  summarize(average_sentiment = mean(overall_sentiment, na.rm = TRUE), .groups = 'drop')

# Explore the average sentiment scores by cancellation policy
head(policy_sentiment)

# Bar plot of average sentiment scores by cancellation policy
ggplot(policy_sentiment, aes(x = cancellation_policy, y = average_sentiment, fill = cancellation_policy)) +
  geom_bar(stat = "identity") +
  labs(title = "Average AFINN Sentiment Scores by Cancellation Policy",
       x = "Cancellation Policy",
       y = "Average Sentiment Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate the x labels for readability

#------------------------------------------------------------------------------------#
#------------------Amenities and price-----------------------------------------------#
#------------------------------------------------------------------------------------#
# `amenities`, and prices
# First, ensure the price is in a numeric format
airbnb_all$price <- as.numeric(gsub("[^0-9.]", "", airbnb_all$price))

# Expand the dataset so that each row represents a single amenity for a listing
amenities_expanded <- airbnb_all %>%
  mutate(amenity = strsplit(gsub("[{}\"]", "", amenities), ",")) %>%
  unnest(amenity) %>%
  group_by(amenity) %>%
  summarise(average_price = mean(price, na.rm = TRUE))

# Filter out empty or irrelevant amenities
amenities_expanded <- filter(amenities_expanded, amenity != "" & !is.na(amenity))
library(wordcloud)
library(RColorBrewer)

# Generate the word cloud, this time using the average price as the frequency
wordcloud(words = amenities_expanded$amenity, freq = amenities_expanded$average_price,
          min.freq = 1, max.words = 200, random.order = FALSE, 
          rot.per = 0.25, colors = brewer.pal(8, "Dark2"))
#----------------------------------------------------------------------------------------------------#
# ----------------------------- STEP 07: Loading the data for Tableau  ----------------------------- #
#----------------------------------------------------------------------------------------------------#

# Define the path where you want to save the JSON file
file_path <- "C:/Users/racha/OneDrive/Desktop/Mban/Business Analysis with Unstructured Data/airbnb_all.json"

# Save the 'airbnb_all' data frame to a JSON file in the specified path
write_json(airbnb_all, path=file_path)



