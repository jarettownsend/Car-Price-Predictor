# 🚗 Car Price Predictor

**Care Price Predictor** is a web app built with Streamlit that predicts the market value of a used car based on features like make, model, year, mileage, and condition. Powered by a trained **Random Forest Regressor**, it gives users quick and reliable pricing estimates drawn from real-world Craigslist listings.

> **Data source:** [Craigslist Cars and Trucks Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

---

## 🚀 Demo

Try it live on **[Streamlit Cloud](https://carvaluepredictor.streamlit.app/)**  

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – for cleaning and manipulating data
- **Scikit-learn** – for model development and training
- **Streamlit** – for deploying the web interface

---

## 🧪 Development

This project involved significant data wrangling and feature engineering before any modeling could begin.

The original dataset included a `model` column that was too messy to use directly. I initially built a version of the model using broader features like `type` (e.g., SUV, sedan) and `size` (e.g., mid-size, full-size), which produced decent results. But that’s not how people typically search for cars. Instead most users think in terms of specific makes and models (like "Toyota Camry" or "Ford F-150").

So I pivoted and used regex-based pattern matching to extract and combine the `make` and `model` fields into a single clean column. This required a lot of manual investigation and iterative cleaning, but it ultimately resulted in a more structured and usable feature representing the most common make-model combinations.

On the modeling side, I started with a basic **Linear Regression**, but it struggled to capture the non-linear patterns in the data. Switching to a **Random Forest Regressor** significantly improved performance and delivered more stable predictions.

I also experimented with **hyperparameter tuning**, but it didn’t yield the gains I expected—likely because the model was already performing near its ceiling, given the noisy nature of the data. In the end, a well-balanced Random Forest model struck the right trade-off between accuracy and complexity.

The final model achieved an **R-squared of 0.9247** and an **RMSE of 3,542.52**, which means that on average, predictions were off by around $3,500. In context, this is a reasonably strong result—for example, if a car is worth $26,000 and the model predicts $29,500, that’s within an acceptable margin for many use cases.


---

## 💡 Potential Improvements

- **Geographic Context**  
  Adding location-based features could help capture regional pricing differences. I originally tried to include `region` as a feature, but it was very messy, and because of how many unique regions there were, it left me with a sparse matrix after encoding. In theory, I could have mapped each region (e.g., "Greensboro", "Florida Keys") to a broader geographic category like "Southeast", but that wasn’t how I wanted to prioritize my time for this project.

- **Better Data**  
  While I’m mostly happy with the dataset I used, it came with some clear limitations. The biggest issue is that the data is over three years old, which is a long time in the used car market. This likely makes my predictions less accurate in today's pricing environment and also makes it difficult to validate results against current listings on platforms like CarGurus or Kelley Blue Book. Additionally, while my regex-based make/model extraction worked reasonably well, having a dataset with pre-cleaned `make` and `model` columns would save a lot of effort and reduce noise.

- **Bayesian Modeling**  
  One future direction I’m interested in is exploring Bayesian approaches to pricing predictions. A Bayesian model could provide not just point estimates, but also **credible intervals** that reflect the uncertainty around each prediction—something particularly valuable in a noisy domain like used car pricing.

  I tried to approximate this kind of uncertainty by calculating a prediction range using the variation across individual trees in the Random Forest. While this gave a rough interval by estimating a standard error, it's ultimately a statistical shortcut. It assumes things like independence and normality that don’t really apply to decision trees, and it lacks the interpretability and theoretical rigor of true Bayesian methods.

  A **Bayesian model** would allow uncertainty to be directly and transparently modeled, offering more meaningful insight into the confidence of each prediction. It’s something I’d like to explore further in future iterations of this project.



