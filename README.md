# Double Bind
### Fighting Trolls and Defending Free Speech in the 2016 Election

![Double Bind Header]()

An incredible aspect of the 2016 presidential election was its capacity to surprise. Nearly 5 years later, and I still find myself in awe of the double bind social media companies found themselves in as result of how the election played out. One narrative that emerged was of Putin the Puppet master hiding deep within our news feeds. Social media companies had become weapons in a new Cold War. On the other hand, in part by trying to address issues of election interference, these same social media companies were accused of choking out free speech at home. And misclassifications of legitimate speech as foreign trolls lead to accusations of bias and disenfranchisement. 

The project seeks to address the double bind grown out of the 2016 election using natural language processing (NLP) and machine learning, to classify trolls while avoiding the misclassifications of legitimate speech. The results will highlight the contradictory nature of this difficult problem and make some suggestions for further research. 

---

### The Data 

To simulate the double bind, I accessed a database of more than 280 million legitimate tweets from [George Washington University](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN&version=3.0) Libraries Dataverse. These ID's were used to access the full text of the tweets using my own script (included in src/) with a huge lift from Documenting the Now's [Hydrator app](https://github.com/DocNow/hydrator). 

For trolls, I forked a [repo](https://github.com/fivethirtyeight/russian-troll-tweets) from [FiveThirtyEight](https://fivethirtyeight.com/features/what-you-found-in-3-million-russian-troll-tweets/) of data obtained from Clemson University researchers Darren Linvill and Patrick Warren. Troll Tweets from FiveThirtyEight were available in full and did not need to be hydrated. 

Both data sets were then filtered to include the text of the tweet, its date and the user. The data sets were then filtered further to include only Tweets between June 28th to November 8, 2016, corresponding to a general overlap in the datasets. Labels indicating if the Tweet was from a troll or legitimate were then added programmatically. 

After this process, I was left with a little more than 5.3 million Tweets. Roughly 7 percent of these Tweets were trolls.

---

### The Exploration

The goal of the modeling phase was to predict the troll/legitimate labels based solely on the text of the Tweet. To get an idea of how the model might perform, I began by exploring word use frequencies among the classes. What I found was that legitimate Tweeters tended to use more generic terms which would be expected to accompany a political contest. Trolls, on the other hand, preferred words associated with divisive social issues. Legitimate Tweets were more likely to contain words such as `election`, `debate`, and `president`, while trolls favored words like `police`, `sports`, and `black`. If you remember the summer of 2016, the Black Lives Matter movement was in full force and that movement had made its way into sports in a public way often personified by the San Francisco 49ers' then-quarterback Colin Kapernick when he began kneeling during the national anthem at NFL games in August of 2016. 

Given this evidence, I beleived there was a reasonable chance that I could begin to classify the Tweets by topic. I count vectorized my corpus, ran a Latent Derichlet Allocation model, and then re-introduced the labels to view the disribution of topics by class. But the model showed no discernable difference in the distribuition of topics. The charts here show five topics, but the results were similarr no matter the number of topics used. Upon reflection, this findinng should hve been such a shock. From what congrresionnal investigations revealed about the troll operation, the goal was not to introduce new topics but to exploit the topics that already existed. This means that even if trolls favored words like `police`, `sports`, and `black`, racial injustice was not a topic unique to them, and distinguishing the classes based on topics was unlikely to succeed. 

---

### The Model 

With topic modeling proving less thann promissing, I pivoted to a random forest classifier. I re-vectorized th corpus, switching from count vectors to TF-IDF vectors. This reintroduced the high dimensionality that I sought to avoid with topic modeling, and since the data contained only 7 percent trolls the training set required rebalancing with SMOTE. Still, even with these challanges, the ROC curve produced by the model showed a way forward. But it also highlighted some of the trade offs that would need to be made. According to the ROC curve, the rate of false positives (i.e. missclassifying legitimate speech as troll speech) grew slowly moving across the curve from left to right. But as the true postive rate (i.e. correctly identifying trolls) reached 50 percent, the rate of false possitives began to grow more rapidly before takeinng a sharp right turn as the true positive rate passed 60 percent. Clearly these inversely related objectives (to increase the rate of correctly identifying trollss while limited the number legitmate speech misclassifications) was going to require compromise and balance to achive the the best model posssible. 
