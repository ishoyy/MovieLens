# Matrix Factorization Recommender System
This project explores collaborative filtering using matrix factorization on the MovieLens 100K dataset. The goal is to learn latent representations of users and movies to generate basic movie recommendations.

## Overview
- The MovieLens 100K dataset contains:
- 100,000 ratings
- 943 users
- 1,682 movies
  
The dataset is transformed into a user–item sparse matrix, where each entry represents a user’s rating for a movie.
Matrix factorization is then applied to decompose this matrix into lower-dimensional representations for users and items.

## Methods
Two matrix factorization approaches were implemented:
Stochastic Gradient Descent (SGD)
- Iteratively updates user and item feature vectors
- Minimizes prediction error between actual and predicted ratings
- Uses regularization to prevent overfitting
2. Alternating Least Squares (ALS)
- Alternates between solving for user and item feature matrices
- Uses closed-form updates via linear algebra
- Provides a different tradeoff between speed and stability
  
Similarity Computation
- After training, cosine similarity is used to measure similarity between movies:

sim(𝐯𝑖,𝐯𝑗)=𝐯𝑖⋅𝐯𝑗‖𝐯𝑖‖‖𝐯𝑗‖
 
​This allows the system to recommend similar movies based on learned feature vectors.

## Experiments
The project evaluates:
Different latent feature sizes:
- k = 10, 30, 50, 100
- Training performance (runtime)
- Prediction error using RMSE (Root Mean Squared Error)
- Recommendation quality based on similarity scores
  
## Example Output
The model generates recommendations by selecting the most similar movies for a given movie ID (0-based indexing).
Example:
```
Top 5 movies similar to movie 0: [  94 1454   70  209  173]
Similarity scores: [0.83752781 0.82875654 0.8286223  0.82041235 0.79572094]
```

## Technologies Used
- Python
- NumPy
- Pandas
- SciPy (sparse matrices)
- scikit-learn (cosine similarity)

## Key Learning Outcomes
Working with large sparse datasets
Implementing matrix factorization techniques
Comparing optimization methods (SGD vs ALS)
Evaluating models using RMSE
Generating recommendations using similarity metrics

## Notes
This project was completed as part of a fundamentals of machine learning course. It focuses on understanding and applying core concepts in recommendation systems rather than production-level optimization.
