# HW5: Linear Regression on Lake Mendota Ice

This project uses linear regression to analyze historical data on Lake Mendota's ice cover, aiming to predict trends and insights about the lake's freezing patterns over time.

---

## Assignment Goals

1. **Data Curation**: Process raw data from 1855-56 to 2022-23 into a clean dataset (`hw5.csv`).
2. **Data Visualization**: Plot trends in ice cover duration.
3. **Data Normalization**: Normalize features to improve gradient descent performance.
4. **Linear Regression**: Solve the regression problem using:
   - Closed-form solution.
   - Gradient descent optimization.
5. **Prediction**: Predict future values and analyze trends.
6. **Model Interpretation**: Discuss the implications and limitations of the model.

---

## Dataset

- **Source**: Wisconsin State Climatology Office.
- **Columns**:
  - `year`: The starting year of the winter season.
  - `days`: Total number of ice days during that winter.

Example (`hw5.csv`):
```csv
year,days
1855,118
1856,151
1857,121

