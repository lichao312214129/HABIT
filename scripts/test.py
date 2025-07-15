import pandas as pd
import re

file = r'H:\results\features\whole_habitat_radiomics.csv'


df = pd.read_csv(file)

# 用正则表达式是第一列的第一个连续数字提取
idx = [re.search(r'\d+', idx_).group() for idx_ in df.iloc[:, 0]]

# Replace the first column with the extracted numeric index
df.iloc[:, 0] = idx

# save the dataframe
df.to_csv(file, index=False)






