# User Interaction Analysis with Articles

## Objective

The objective of this code product is to provide a solution for newspaper companies that have access to user preferences information. By selling this code product to companies like Nike, who have a limited budget for advertising, the code can help them effectively target a specific audience. The code will return a list of the top `n` users who are interested in a specific subject, allowing the company to display advertisements more efficiently. Additionally, the newspaper company can benefit from monetizing the information they hold.

## Overview

This project analyzes user interactions with various articles, categorizing them into four main areas: Sports (運動), Arts (藝文), Finance (財經), and Politics (政治). The goal is to determine each user's preference for these categories based on their interactions with the articles.

## Process

- Data Collection: Gather user interaction data with articles (Articles A to P) and record the number of interactions per user.

- Similarity Scores: Assign similarity scores to each article for the four categories. These scores represent how closely an article relates to a given category.

- User Score Calculation: Calculate the total score for each user in each category by multiplying the number of interactions with the similarity scores.
  Apply a softmax normalization to these scores to understand the user's preference distribution across the categories.

- Analysis: Identify the user with the highest normalized score in each category. This reveals the user who shows the most interest in that category.

- Results: The program outputs the user with the highest score in each category along with their score, providing insights into user preferences.

## Output

The final output includes:

1. Similarity scores for each article across the four categories.
2. Total and normalized scores for each user.
3. The user with the highest preference in each category.

## Example Output

```sql
Similarity Scores: {...}
User Scores: {...}
Softmax Normalized Scores: {...}
Max value for '運動': Luna with score 0.2451
Max value for '美術': Ray with score 0.2636
Max value for '財經': Eva with score 0.2615
Max value for '政治': Bob with score 0.2366

```
