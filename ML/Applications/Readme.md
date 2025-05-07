### How to load a Data Set

We can load a dataset with the help of read_csv function provided in the pandas library. Pandas is a Python library used for data analysis and manipulation, providing efficient data structures like DataFrames to handle structured data.

```
import pandas as pd

dataSet_1 = pd.read_csv("oecd_bli_2015.csv", thousands=',')
dataSet_2 = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")
```

The parameters padded to the read_csv function are very important, as they properly structure the data for training the model and predicting new values.

1. thousands=',' → Converts "1,234" into 1234.
2. delimiter='\t' → Reads tab-separated files (TSV). Uses a tab (\t) as the separator instead of a comma.
3. na_values="n/a" → Automatically converts "n/a" into NaN.
4. encoding='latin1' → Supports special characters.

Features are the columns with which we predict the dependent variable.

We should not perform feature scaling before the train test split because if we do that then the model can identify the test set that we are securing from training, which should not happen. In simple terms data lekage of test set will happen.
