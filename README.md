# Calorie Prediction using Python/Sci-Kit Learn
The first part of this project, including exploratory data analysis, can be found [here](https://aspokasjaros.github.io/recipe_analysis/)
---
## Framing the Problem
My goal here is to predict the total calories of a recipe given its nutritional info using a form of linear regression. By modeling each macronutrient's role in total calories, being able to find a specific recipe with a focus on a specific macronutrient, but also the total number of calories, will be considerably easier for anybody who comes across this report. This would be beneficial for people like myself; athletes who are constantly looking for filling meals that can provide lots of energy/calories but are also high in proteins to aid muscle building. Students who live on campus, where all the dining hall meals are high in sodium and sugar, can better understand how much of their calories are attributed to nutrients that won't provide enough energy to get through their academically rigorous day. At the time of prediction, I will be using percent daily value (PDV) of total fat, sugar, sodium, protein, saturated fat, and carbohydrates to predict number of calories.

The response variable I am predicting is total number of calories. Personally, I often pick my recipes to cook for the week by the proprtion of macronutrients (and also what is on sale at the grocery store), so being able to interpret a probable caloric intake based on what macronutrient I choose to focus my meals around will personally benefit me, and likely others as well. I chose to use linear regression here because, logically, the calories of a recipe will increase the more macronutrients it has total, but I wanted to better understand the relationship of how each macronutrient contributes to the overall count. The metrics I will be tracking is root mean squared error (RMSE) because it is easily interpretable by the average person (the non-coder). In this report, RMSE represents the average number of calories my prediction differed from the actual value by. The other response variable I will be using intermittently is R^2, a measure of fit for regression models where values closer to 0 mean the model doesn't represent the data generating process very well, while values closer to 1 do. I will use R^2 to quantify the improvement of my model over time.

---

## Baseline Model
All features used in both the baseline and final models will be quantitative. For the baseline model, I will be using sodium (PDV) and sugar (PDV). My variables were already available as floats, so no encoding was necessary.

The baseline model is a standard Linear Regression from Sci-Kit Learn. This regression model takes in columns corresponding to my chosen feature parameters (sodium and sugar) and a column of actual values for the response variable (calories), and is able to interpret weights/slopes for the parameters and an intercept.

In the baseline model, the intercept is 274.67, weight of sodium is 0.88, and the weight of sugar is 1.89. Because of how large the intercept is, I am suspicious that sodium and sugar are not mostly reponsible for the amount of calories in most recipes. I can check this by reviewing the R^2 for this model by using the .score method which takes in the feature parameter columns and the response variable column. The returned number is approximately 0.46, which confirms my suspicions and implies that this model is not fully capturing the data generating process. I was curious, however, how well my model was predicting calories. Using Sci-Kit Learn's mean_squared_error method and taking the square root of the value outputted by that, I was able to see that my RMSE is approximately 20 calories. The fact that this value is so low is surprising, considering how low the R^2 is.

One of the most important aspects of modelling information like this, is the model's ability to generalize to unseen data. To test this in my model, I used Sci-Kit Learn's train_test_split model which takes in the feature parameter columns as X, the respoonse varable column as y, and a test_size as a proportion of values that will be in the test split. To understand how well my model works with unseen data, I can train the model on the train split and compare the R^2 and and RMSE. On my training set, the R^2 was 0.4768 and the RMSE was 20.67, and on the test set the R^2 was 0.4563 and the RMSE wsa 20.26. With these values, I can determine that the model is not overfitting the training data and can generalize well to unseen data.

___

## Final Model
To improve on my baseline model, I also wanted to consider how other macronutrients impact total calories, so I added the columns for total fat, protein, saturated fat, and carbohydrates to my feature parameters. After further consideration, it is likely more calories come from fat, protein, and carbohydrates as these are the main building blocks for energy that people need to fuel their bodies.

Since my goal for this final model is to predict total calories as accurately as possible, I decided to use SciKit Learn's GridSearchCV to test multiple alpha parameters for multiple types of linear regression. Specifically, I test for Ridge Regression and Lasso Regression, each with their own tests of alpha values of 0.1,1.0, and 10. (note: Linear Regression does not take any alpha values). The result of GridSearchCV was to use Lasso Regression with alpha=0.1.

Lasso Regression works to minimze the sum of squared difference between predicted values and actual values, similar to most regressions. Where Lasso Regression differs, however, is that it penalizes smaller coefficents as the size of the data increases in order to make the weight of practically neligible parameters basically 0 (or even negative as we'll see shortly).

Since this is a new model, I needed to ensure that it can still generalize to unseen data. I used train_test_split again with the same test proportion and used both data sets to ensure that my model isn't overfitting and can provide plausible prediction for calories based on given nutritional information. After fitting the new model to the data, the intercept is considerably smaller at 11.25 (implying that a recipe with 0 PDV for each macronutrient tracked will have about 11 calories). The new weights provide a completely different perspective on the role of sugar and sodium in recipes for me (I had previously assumed sugar contributed far more heavily, hence why I used it as one of my baseline model parameters) but the weights for sugar and sodium were 0.00 (so small that Python had to round it down to 0) and -3.85*e^-3 respectively. Sodium has such a small contribution to total calories that it actually "takes away" from the final value. The parameters with the highest weights were carbohydrates (11.65), total fat (5.69), and protein (2.16). The returned weights align with my prediction before modelling.

At this point, I want to know if I have improved my model or if I have overcomplicated and need to change my hyperparameters or decrease the number of features I used. To do this, I found the RMSE and R^2 of my training and test sets.
|Data Set| R^2 | RMSE |
| ----------- | ----------- |----------- |
| Training | 0.9956 |  6.22 |
| Test | 0.9960 |  5.99 |
<table>
    <tr>
        <th>Data Set</th>
        <th>R^2</th>
        <th>RMSE</th>
    </tr>
    <tr>
        <td>Training</td>
        <td>0.9956</td>
        <td>6.22</td>
    </tr>
    <tr>
          <td>Test</td>
          <td>0.9960</td>
          <td>5.99</td>
      </tr>
</table>

From the table above, we can see the R^2 for both the training set and test set are incredibly similar, implying the model is not overfit to the training data, and both very close to 1, implying the model accurately captures the data generating process and can explain the different variances from the means of the data set. In the RMSE column, the numbers are slightly different still, but have significantly improved since the baseline model. This tells me the model can more accurately predict total calories of a recipe based on given macronutrient infomation with an average error of about 6 calories.

---

## Fairness Analysis
The final aspect of my model I want to confirm is that it is a fair representation of all data, regardless of any possible outlying factors. To test this, I will split the dataset in two halves: "bad recipes" where the average rating is strictly less than 3.0, and "good recipes" where the average rating is greater than or equal to 3.0. I chose this split because I've recognized a trend around me where people tend to praise "healthier" recipes considerably more, and would probably give these recipes a better rating than "bad" recipes that are not as "healthy." People often judge a recipe's "health" by it's total calories, where high-calorie recipes are deemed unhealthy. With these assumptions, my null hypothesis is that my model is fair for both "healthy, good" recipes and "unhealthy, bad" recipes. The alternative hypothesis is the model can more accurately predict "healthy" recipes due to the observed calorie value for those recipes being lower, so the RMSE would also likely be lower.

After splitting my data into two data sets with the comparison described above, I re-fit and ran my final model on the "good" and "bad" data sets and got the following metrics:
|Data Set| R^2 | RMSE |
| ----------- | ----------- |----------- |
| Good | 0.9957 |  6.13 |
| Bad | 0.9989 |  4.68 |

All the metrics are close enough to the original metrics from the final model that I fail to reject the null hypothesis. What is interesting to me, however, is that the error for the "bad" recipes is smaller than the error for "good" recipes, which contradicts what I predicted I may see in the paragraph above. With the given information, I can confidently say my model is highly likely to be fair for recipes of all ratings.

---
