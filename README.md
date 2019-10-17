# Recommand System using Collaborate Filtering

## Introduction

The personalized recommendation system, based on a collaborative filtering algorithm (CF), has been successfully applied in the Internet field, especially in the aspects of e-commerce and advertising business. User-based CF, item-based CF, are traditional common collaborative filtering algorithm.

There are three different collaborative filtering algorithms in this repository: User-based CF, Item-based CF; and Matrix Factorization Latent Factor model. Their performance are evaluated and compared by RMSE.

## Dataset Description

In this project, the data set is from the Grouplens site with 100,836 ratings applied to 9,742 movies by 610 users. 

## Train-test Set Split

To evaluate the performance of the algorithm, we have to split the data into the train set and test set.

The method is based on users, which means that for each user, we split the movies he watched with a fixed portion, which is 0.8 in this experiment, one into the train set and the other into the test set. 

After the train-test split, the train set and test set both contain all users. We use the train set to predict every possible user-movie combination's rating and compare it with the test set.

## Evaluation Method

This project uses cross-validation to prevent over-fitting, which may be caused by overly complex models and evaluate by Root Mean Squared Error (RMSE) to measure the model accuracy.

1. K-fold cross validation

   K-fold cross-validation will let all samples used as a training set and test set, which means each sample is verified once. In this project, we choose K = 5. Cross-validation is repeated for 5 times, each time a subset is selected as the test set, and the average recognition accuracy of cross-validation for 5 times is taken as the result.

2. Root Mean Squared Error (RMSE)

   Root Mean Squared Error (RMSE) is a popular measurement method and its equation is as $(1)$.

   $T$ is the whole number of ratings in the test dataset, is the ground truth, denotes the predicted ratings for $T$.

$$
RMSE = \sqrt{\frac{1}{|T|}\sum\limits_{(u, i)\in T}(\hat{r}_{ui} - r_{ui})^2} \tag{1}
$$

## Collaborative Filtering

Collaborative Filtering algorithm is the most popular type of recommendation algorithm, which has successfully applied in a variety of areas, like Netflix, Youtube, e-bay and so on. 

The advantage of collaborative Filtering algorithm is that it does not require abundant domain-specific knowledge and uncomplicated engineering implementation. User-based collaborative filtering, item-based collaborative filtering, and model-based collaborative filtering are three types of collaborative filtering recommendations. Among them, User-based CF and item-based CF are on memory-based. Different Collaborative Filtering algorithm can be adapted to a variety of recommendation system requirements. Usually, model-based collaborative filtering is the most popular type of collaborative filtering recommendation system.

## User-based

### Theory

The key idea of User-based collaborative filtering is that we assume users can give valuable advice to another user who is very similar to them. So we can gather the “opinions” from other similar users to give recommendation suggestions to our target user.

Our goal has two parts, one is to evaluate a rating for some specific user-movie combination. For example, we have user $a$ and movie $m$ and we need to predict what rating will user $a$ gives to movie $m$. The other is to recommend some movies to users.

So how to do that? According to the user-based collaborative filtering theory, we first need to find all users who watched movie M, name them as a set U~m~. Then we need to find the top K (K is an integer used to decide how many similar users we needed to make a prediction) most similar users other than A himself in the list U~m~, name them as another set U~mKa~, sometimes the number of users in U~m~ might less than K, so in such condition, we will use all these users as voters. Finally, we calculate the prediction based on these top K users' ratings, r~mu~ and their similarity with $a$, using the formula below:
$$
p(a, m) = \sum\limits_{u\in{U_{mKa}}}{\frac{sim(a, u) * r_{mu}}{\sum\limits_{v\in{U_{mKa}}}{sim(a,v)}}} \tag{2}
$$
$sim(A, u)$ is the similarity between user $a$ and $u$.

And if we want to recommend some movies to a user, the recommend score for a movie $m$ considering user $a$ is calculate by the formula below:
$$
score(a, m) = \sum\limits_{u\in{U_{mKa}}}{sim(a, u) * r_{mu}} \tag{3}
$$
So how to evaluate the similarity between two users? This is the core of the User-based collaborative filtering. It is natural to think that two users are very similar if they have a lot of common movies watching history. On the other hand, they should have similar tastes if they have close ratings on the same movie.

Based on the ideas above, the similarity between the user $u$ and $v$ is evaluated by the formula below:
$$
sim(u, v) = \frac{\sum\limits_{m\in C(u,v)}\frac{5-|r_{mu} - r_{mv}|}{ln(1 + |C(u,v)|)}}{\sqrt{|N(u)| * |N(v)|}} \tag{4}
$$
$N(u)$ is the set of movies watched by $u$, $C(u,v)$ is the set of movies both watched by $u$ and $v$.

The $5 - |r_{mu}-r_{mv}|$ is to evaluate if two users' ratings to a same movie are close. For example, if both users give the same rating, it will turn out 5, if one user gives 5 and the other gives 1, it will turn out 1.

The $ln(1+ |C(u, v)|)$ in this formula is to punish the movie which is very popular.

### Implementation

The training process may take up to 30 minutes to finish.

#### Calculate the similarity matrix

According to the similarity formula mentioned above, some components we need to get are $N(u)$ for every user u and $C(u, v)$ for every (u, v) combination.

It is easy to get $N(u)$, we just need to group the data by user, and find the length of each group.

To get $C(u,v)$, we just need to intesect $N(u)$ and $N(v)$.

Finally put them into the similarity formula and get the result.

#### Performance

We use cross validation to test this recommendation system's performance and here is the result:

Mean RMSE: 0.954, variance: 2.21e-5.

From the result, I think the User-based recommend system does a good job and has a very stable performance.

#### RMSE on different K values

We also tried different K values to see if there will be any difference.

Here is the result:

![user_based_K](./resource/user_based_K.png)

<center>figure.1 the K-RMSE curve of user-based</center>
Same as our anticipation, with the growing of K value, the RMSE error drops rapidly at first, then decreases gently. 

## Item-based

In this part, we will try to use the Item-based Collaborative filtering algorithm to predict scores for a set of unwatched movies by a specific user and recommend some movie to this user. Unlike the User-based CF algorithm, the item-based method looks into the connections between the set of movies the target user has rated and the movies the target user has not interacted yet. This approach compute the similarity between a target movie $j$ and selected $k$ most similar movies $\{i_1,i_2,...,i_k\}$ the target user has rated. By having the corresponding similarities $\{s_{j,1},s_{j,2},...,s_{j,k}\}$ , the prediction is computed as a weight average score to the target movie $ j$.

### Theory

The key assumption behind this algorithm is that we assume users will always remain their taste in movies.

This whole modelling process can be divided into two steps, namely, Similarity Calculation and Prediction. The similarity computation is the most critical move in the entire modelling which determines if the final results are satisfying or not and is pre-computed beforehand.

#### Basic idea of similarity computation

To compute the similarity of two given movie $i$ and $j$ , by only looking into the co-rated users, a co-occurrence matrix $C$ is first constructed, and each entry in the matrix records the number of users who like both movie $i$ and movie $j$. More details will be explaned in the figure.2 (Badrul S et al, 2001, p.6) and figure.3 (Liang X, 2012, p.):

![Screen Shot 2019-08-09 at 10.59.55 pm](./resource/3.png)

<center>figure.2</center>
The figure1 represents the process of selecting co-rated users for movie $i$ and movie $j$. The columns are the movies and the rows are users.

![Screen Shot 2019-08-09 at 11.28.36 pm](./resource/4.png)

<center>figure.3</center>
Figure.3 elaborates how the co-occurrence matrix is constructed. Each row in the leftmost part indicates movies of a user, each entry in the matrix in the rightmost part implies the total number of users who interacted with both movie $i$ and movie $j$ .

In the computation of similarities between items, an adjusted cosine similarity method and a weight sum method are applied. The reason to use adjusted cosine instead of basic cosine is that it can offset the drawback of the existing fact that different users have a different rating scale. The specific formula applied is as below:
$$
sim(i,j) = \frac{\sum_{u\in U}(R_{u,i}-\overline{R})(R_{u,j}-\overline{R})}{\sqrt{\sum_{u\in U}(R_{u,i}-\overline{R})^2}\sqrt{\sum_{u\in U}(R_{u,j}-\overline{R})^2}}\tag{5}
$$
where$  \overline{R}$ denotes the average ratings rated by user $u$ .
$$
P_{u,j} = \frac{\sum_{i \in N}S_{i,j}\times R_{u,i}}{\sum_{i \in N}|S_{i,j}|}\tag{6}
$$
where $j$ is the target movie for prediction, $u$ is the target user and $N$ is the set of movies that user $u$ has rated.

### Implementation

#### Performance

A cross-validation is also applied for the measurement of accuracy of this predictive model, and an average RMSE of  0.863 and a variance of 1.13e-4 are obtained. This indicates the model is very steadible.

#### RMSE on different K values

After the experiments on a single target user, the scope is changed towards the performance of average RMSE on all users in terms of a range of K values. 

![img](data:image/png;base64,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)

<center>figure.4</center>
The figure indicates a rapid decrease when the K values increase in the starting point, and when the K values get larger, the average RMSE tends to be steadible as anticipation.

### Insights & Improvements

While experimenting, some unignorable problems are encountered. 

First, popular movies seem to have a higher similarity with other movies, which leads to a higher prediction. Eventually, the model will tend to recommend the popular movie but not necessarily be interested in the user. Therefore, to tackle this problem, a normalization on the weights is applied when computing the prediction. This improves the average RMSE to 0.891 with a K value of 10.

Another problem is that this algorithm can not make predictions on those movies that nobody has ever rated. Then the unpopular but high-quality movies are unlikely to be recommended before anyone becomes the first person who eats crab.

## Matrix Factorization Latent Factor Model

### Theory

The matrix factorization has an assumption that a set of common attributes exists for all items, also each user has their rating for each of these attributes, independent of items. These attributes are hidden and are called latent factors. One strength of matrix factorization is that it allows incorporation of more information, if explicit information is not available, the recommender system can still infer some implicit information and make use of it. This method also has a good extension performance. 

The key idea of matrix factorization collaborative filtering is to transform the matrix to represent latent factors, the missing value in the original matrix will be calculated by the decomposed matrix.
$$
R_{m×n}≈ X_{m×K}*Y_{K×n} = \hat{R}  \tag{7}
$$
where **K** is an intermediate variable representing k latent factors, matrix **X(m,K)** represents the relationship between **m** movies and **K** features, matrix **Y(n,K)** represents the relationship between **n** users and **K** features, here we use cross-validation for choosing the optimal **K**. 

![2](./resource/2.jpg)

<center>figure.5 Approximating the ratings matrix with row and column factors</center>
This is called a low-rank approximation in linear algebra, which compressing the sparse data in R into lower dimension spaces **m×K** and **K×n**. The multiplication of **X** and **Y** would output a matrix **$\hat{R}$**,  which is an approximation of original matrix **R**.

1. To calculate the rating of user *u* for item *i*,  take the dot product of the two vectors:

$$
\hat{r}_{ui} = \mathbf{x}^{T}_{u} \cdot \mathbf{y}_{i} = \sum_{k=1}^{K}\mathbf{x}_{ik}\mathbf{y}_{kj} \tag{8}
$$
2. Add regularization terms to loss function to avoid overfitting. Loss function:

$$
L = \sum_{u,i}(r_{ui} - \mathbf{x}^{T}_{u} \cdot \mathbf{y}_{i})^{2} + \lambda\sum_{u}\left\lVert\mathbf{x}_{u}\right\rVert^{2} + \lambda\sum_{i}\left\lVert\mathbf{y}_{i}\right\rVert^{2} \tag{9}
$$
Here the constant $\lambda$ is controls the regularization is also determined by cross-validation.

To minimize the loss function, we use Stochastic Gradient Descent (SGD) to loop through all ratings in the train set and compute the associated prediction error.
$$
e_{ui} = r_{ui} - \mathbf{x}^{T}_{u} \cdot \mathbf{y}_{i} \tag{10}
$$
3. Obtain update rules for $\bold x_u$, $\bold y_i$, firstly calculate the negative gradient of the loss function

$$
\frac{\partial}{\partial x_{i, k}} E_{i, j}^{2}=-2\left(r_{i, j}-\sum_{k=1}^{K} x_{i, k} y_{k, j}\right) y_{k, j}+\beta x_{i, k}=-2 e_{i, j} y_{k, j}+\beta x_{i, k} \tag{11}
$$

$$
\frac{\partial}{\partial y_{k, j}} E_{i, j}^{2}=-2\left(r_{i, j}-\sum_{k=1}^{K} x_{i, k} y_{k, j}\right) x_{i, k}+\beta y_{k, j}=-2 e_{i, j} x_{i, k}+\beta y_{k, j} \tag{12}
$$

then, update the matrix components according the the direction of the negative gradients:
$$
x_{i, k}^{\prime}=x_{i, k}-\alpha\left(\frac{\partial}{\partial x_{i, k}} e_{i, j}^{2}+\beta x_{i, k}\right)=x_{i, k}+\alpha\left(2 e_{i, j} y_{k, j}-\beta x_{i, k}\right) \tag{13}
$$

$$
y_{k, j}^{\prime}=y_{k, j}-\alpha\left(\frac{\partial}{\partial y_{k, j}} e_{i, j}^{2}+\beta y_{k, j}\right)=y_{k, j}+\alpha\left(2 e_{i, j} x_{i, k}-\beta y_{k, j}\right) \tag{14}
$$

4. Continue the iteration until the algorithm converge	

5. Prediction

After getting **X(m,K)** and  **Y(K,n)** matrices, use them for making predictions for user **i** and item **j**.
$$
\sum_{k=1}^{K} x_{i, k} y_{k, j} \tag{15}
$$
Overall, the Matrix Factorization Recommender System learns the model by fitting the previously observed ratings and to generalize a way that could predict future, unknown ratings. 

### Performance

We trained a model using the gradient descent optimizer, with learning rate initialized to 0.001, and the regularization parameter $\lambda$  is set to  0.7 as a cross-validation result. The model is built with 11 factors, with gradient descent running for 100 iterations. With this setting we obtained the following:

Mean train RMSE: 0.763, Mean test RMSE: 0.961

### Insights & Improvements

The problem of Latent Factor Model is that it is hard to implement real-time recommendation since it will incorporate all users' records to calculate the factorized matrices to represent the relationships between latent factors and users/items, also this model requires repeated iterations to achieve better performance, which is of high time complexity. 

This model can be improved by incorporating both user and item biases, also it can be improved by choosing better parameters of regularization, parameters too low could cause overfitting, and too high will lead to large prediction error.


