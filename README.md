# Double Bind: Fighting Trolls and Defending Free Speech in the 2016 Election


<p align="center">
  <img width="800" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/double_bind_header.png">
</p>


A lingering aspect of the 2016 presidential election is its capacity to surprise. Nearly 5 years later, and I still find myself in awe of the double bind social media companies found themselves in as result of how that election played out. One narrative that emerged after the polls closed was of Putin the Puppet Master hiding deep within our news feeds. Social media companies had become weapons in a new Cold War. On the other hand, in part by trying to address issues of election interference, these same social media companies were accused of choking out free speech at home, and misclassifications of legitimate speech as foreign trolling lead to accusations of bias and disenfranchisement. 

This project seeks to address the double bind grown out of the 2016 election using natural language processing (NLP) and machine learning to classify trolls while avoiding the misclassifications of legitimate speech. The results will highlight the contradictory nature of this difficult problem and make some suggestions for further research.  

<p align="center">
  <img width="600" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/poltical_cartoons.png">
</p>


---

### The Data 

<img align="right" width="250" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/data_logos.png">

To simulate the double bind, I accessed a database of more than 280 million legitimate tweets from [George Washington University](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDI7IN&version=3.0) Libraries Dataverse. These IDs were used to access the full text of the tweets using my own script (included in src/) with a huge lift from Documenting the Now's [Hydrator app](https://github.com/DocNow/hydrator). 

For trolls, I forked a [repo](https://github.com/fivethirtyeight/russian-troll-tweets) from [FiveThirtyEight](https://fivethirtyeight.com/features/what-you-found-in-3-million-russian-troll-tweets/) of data obtained from Clemson University researchers Darren Linvill and Patrick Warren. Troll Tweets from FiveThirtyEight were available in full and did not need to be hydrated. 

Both data sets were then filtered to include the text of the tweet, its date and the user. The data sets were then filtered further to include only Tweets between June 28th to November 8, 2016, corresponding to a general overlap in the datasets. Labels indicating if the Tweet was from a troll or legitimate were then added programmatically. 

After this process, I was left with a little more than 5.3 million Tweets. Roughly 7 percent of these Tweets were trolls.

---

### The Exploration

<img align="right" width="400" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/word_use_freq.png">

The goal of the modeling phase was to predict the troll/legitimate labels based solely on the Tweet's text. To get an idea of how the model might perform, I began by exploring word use frequencies among the classes. What I found was that legitimate Tweeters tended to use more generic terms associated with a political contest, including words such as `election`, `debate`, and `president`. Trolls, on the other hand, preferred words associated with divisive social issues like `police`, `sports`, and `black`. If you remember the summer of 2016, the Black Lives Matter movement was in full force and that movement had made its way into sports in a public way, often personified by the San Francisco 49ers' then-quarterback Colin Kapernick when he began kneeling during the national anthem at NFL games in August of 2016. 

Given this evidence, I believed there was a reasonable chance I could begin to classify the Tweets by topic. I count vectorized my corpus, ran a Latent Dirichlet Allocation model, and then re-introduced the labels to view the distribution of topics by class. But the model showed no discernible difference in the distribution of topics among each class. The charts below show the results with five topics, but the models were similar no matter the number of topics used. Upon reflection, this finding shouldn't have been such a shock. From results of congressional investigations into the troll operation, we know the goal was not to introduce new topics but to exploit topics that already existed. This meant that even if trolls favored words like `police`, `sports`, and `black`, racial injustice was not a topic unique to them--even if they were more likely to poke at the topic--and distinguishing the classes based on topics was unlikely to succeed. 

<p align="center">
  <img width="700" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/topic_model.png">
</p>

---

### The Model 

<img align="right" width="400" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/roc_curve.png">

With topic modeling proving less than promising, I pivoted to a random forest classifier. I re-vectorized the corpus, switching from count vectors to TF-IDF vectors. This reintroduced the high dimensionality that I sought to avoid with topic modeling, and since the data contained only 7 percent trolls the training set required rebalancing with SMOTE. Still, even with these challenges, the ROC curve produced by the model showed a way forward. But it also highlighted some of the trade offs that would be required. According to the ROC curve, the rate of false positives (i.e. misclassifying legitimate speech as troll speech) grew slowly moving across the curve from left to right. But as the true positive rate (i.e. correctly identifying trolls) reached roughly 50 percent, the rate of false positives began to grow more rapidly before taking a sharp right turn as the true positive rate passed 60 percent. Clearly optimizing these inversely related objectives (to increase the rate of correctly identifying trolls while limiting the number legitimate Tweet misclassifications) was going to require compromise and balance to achieve the the best model possible.  

<img align="right" width="375" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/balanced_mod_CM.png">

To reach this balance, eventually a threshold of 57.95 percent was was set. Precision measured at 32.3 percent and recall at 74.3. This resulted in only 11.8 percent of all legitimate Tweets being misclassified (false positives). It also misclassified 25.7 percent of all trolls. While this may not seem ideal (which no compromise is) when looking at the raw numbers from the test set, you can see the importance of keeping the the percentage of false positives as low as possible. For this balanced model, 11.8 percent of legitimate Tweets in the test set equated to 147,488 Tweets. This is compared to 24,416 misclassified troll Tweets. The bind here is that the more the model pushes the percentage of false positives down, the higher the percentage of trolls that go undetected. 

To illustrate this point, I raised the threshold to 75 percent. This greatly improved the precision score to 94.7 percent. It also reduced the percent of legitimate tweets misclassified to a near perfect 0.1 percent. But the misclassification of troll Tweets jumped to 67.9 percent with a recall score of 32.1 percent. There is an argument to be made here that this model is no better than no model at all, and the original problem of identifying troll Tweets is hardly being addressed. 

On the flip side, a threshold of 25 percent does an excellent job of identifying trolls. The model's recall score and percentage of trolls correctly classified jumps to 97.2 percent. The rub is that everyone is now a suspect, and slightly more than half of legitimate Tweets are misclassified. This dragged down the precision score to a lousy 12.5 percent. 

<p align="center">
  <img width="600" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/CM_7525.png">
</p>


---

### The Evaluation

Looking at the balanced model's Gini importance, many of the same words that appeared in the word frequency chart appear there as well. `Election`, `debate`, and `police` top the list, but `president` and `sports` also make appearances among the top-ten important words. `World` is another word that appears as fifth on the Gini importance and also appeared in other iterations of the word frequency charts as a term favored by the trolls. My hypothesis is that similar to racial inequality, the trolls began to push the issue of globalization as it too became a legitimate, if divisive, issue in the election. Although more investigation to confirm or deny this hypothesis would be required. 

While these words and the differences in their frequency of use proved important to classifying the Tweets, the presence of a word did not guarantee a correct classification. Below are two Tweets about the police that the model misclassified. Both Tweets could be read as right-wing, pro-police, and at best diminish, if not glorify, violence. It's easy to see why the model would have trouble distinguishing them. Although it is interesting to note the Tweets come from different classes and for each class the model classified incorrectly. See if you can correctly identify them. 


<p align="center">
  <img width="800" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/tweet_quotes.png">
</p>

*(If you guessed troll on the left and legitimate on the right, you've beat the model!)*

---

### The Future

It's clear that further investigation into the misclassifications of both trolls and legitimate Tweets is warranted. Discoveries there may go a long way to improving the model and would be my immediate next step. A quick pass of only the misclassifications through the model with a 50 percent threshold showed decent results, but I suspect a closer look, particularly at the Gini importance, could improve the results further and provide more insights into the dataset.

Additionally, given the difficulty of balancing misclassification types, the severity of the penalty for misclassification for Twitter, and the difficulty of correctly identifying Tweets like the two above, other models should be considered. Perhaps newer NLP techniques such Bidirectional Encoder Representations from Transformers (BERT) could be leveraged to better understand a Tweet's context.  

---

### The Tech

<p align="center">
  <img width="500" src="https://github.com/leckieje/electiontrolls_2016/blob/main/img/tech_stack.png">
</p>


#### Contact:

leckieje@gmail.com

*Photo credit: [Mark Peterson](https://www.instagram.com/markpetersonpixs/?hl=en)
