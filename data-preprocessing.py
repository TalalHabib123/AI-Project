import pandas as pd 

data=pd.read_csv('metadata.csv')

MEL = data[data['MEL']==1]  # 0
NV = data[data['NV']==1]  # 1
BCC = data[data['BCC']==1] # 2
AKIEC = data[data['AKIEC']==1] # 3
BKL = data[data['BKL']==1] # 4
DF = data[data['DF']==1] # 5
VASC = data[data['VASC']==1] # 6

# Create an empty DataFrame
new_data = pd.DataFrame(columns=['image', 'label'])

# List of variables and corresponding labels
variables = [MEL, NV, BCC, AKIEC, BKL, DF, VASC]
labels = [0, 1, 2, 3, 4, 5, 6]

# For each variable and label
for var, label in zip(variables, labels):
    # Extract the 'image' column and assign the label
    temp_data = var['image'].to_frame()
    temp_data['label'] = label
    
     # Append to the new DataFrame
    new_data = pd.concat([new_data, temp_data], ignore_index=True)


new_data.to_csv('processed_data.csv', index=False)