# Facebook Marketing Plan for Movies

## Introduction & Executive Summary
### Project Idea
Movie production is undoubtedly one of the most expensive and risky investments in today’s world. Producers would make every effort to secure a higher gross revenue. The entertainment industry goes hand in hand with the social media platforms, hence it is vital for companies to understand how to utilize the tools to maintain its presence in the social media platforms e.g. Facebook to promote their movies, engage with their supporters and ultimately boosting the box office revenues. As a marketing company, we assumed a movie producer came to us and requested for an effective digital marketing strategy on Facebook for their multimillion movie productions. Our goal is to analyze the marketing behaviors of previously released films and design an effective Facebook marketing activities for various types of movies to increase user engagement and maximize the box office revenue.

### Data and Methodology
The data we used in this project are the 5 datasets supplied by the client namely ‘boxoffice’, ‘fandango_review’, ‘fbcomments’, ‘fbposts’ and ‘imdb_movie_overview’. It contains the characteristics of 203 movies, their promotional behaviors on Facebook and respective responses from audience. To tackle the problem, we performed a 5 step analysis as below:

1. EDA, Data Pre-processing and Feature Engineering: To explore the movie data, summarize key characteristics and patterns and preprocess the data
2. Movie Page Segmentation: Perform a 2-step clustering methods. Group movies according to their features and identify clusters that have similar promotional activities
3. Marketing Performance Analysis: Find out what kinds of Facebook marketing would lead to higher box office revenue. Build machine learning models, use RMSE as evaluation criteria and find out the determinant variables using feature importance
4. User Engagement Analysis: Identify the kinds of user engagement that would result in a higher box office revenue. Conduct text mining on the Facebook comments to see how movie fans react to these promotions
5. Insights and Recommendations: Suggest the suitable marketing plan for different clusters to be adopted by the movie makers

### Key Findings and Results
Our goal is to provide a solution to the film producers by first identifying the cluster that their movies belong to and suggest corresponding marketing plan for their movies on Facebook and the user engagement goals that are optimal in generating a higher gross revenue.
By conducting the clustering analysis, 3 clusters were identified based on the Genres. We labelled them as Intense Movies, Relaxing Movies and Humanities & Art Movies. We understood that the marketing plan and user engagement might be different for various type of movies. Hence, we further performed a sub-clustering analysis to discover 5 important features (promotion start time, average no. of posts before debut, post type, timing to post and post format) and provided some intuitions for further analysis.
We identified the important features for each category and derived a recommendation plan for the 3 clusters of movies respectively. The suggestions are based on 3 dimensions of time i.e. before movie release, while movie on shows and movie off shows. Meanwhile, we also found out the significant features and provide recommendations on the user engagements that the company should put effort to build for respective cluster. 
