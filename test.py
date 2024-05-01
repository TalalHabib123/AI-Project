# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv('predictions.csv')

# fig , ax = plt.subplots(2, 1, figsize=(10, 11))
# success_rate = df[df['Predicted Class'] == df['Actual Label']].groupby('Actual Label').size() / df.groupby('Actual Label').size()
# diseases = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
# success_rate.index = success_rate.index.map(lambda x: diseases[x])

# # Use a color palette and add the count above each bar
# sns.barplot(x=success_rate.index, y=success_rate.values, hue=success_rate.index, ax=ax[0], palette='viridis')
# for i, v in enumerate(success_rate.values):
#     ax[0].text(i, v + 0.01, str(round(v, 2)), color='black', ha='center')
# ax[0].set_title('Probability of Successful Prediction for Each Label')
# ax[0].set_ylabel('Success Rate')
# ax[0].set_xlabel('Disease')

# total_count = df.groupby('Actual Label').size()
# successful_count = df[df['Predicted Class'] == df['Actual Label']].groupby('Actual Label').size()

# count_df = pd.DataFrame({'Total Count': total_count, 'Successful Prediction Count': successful_count}).reset_index()

# melted_df = count_df.melt(id_vars='Actual Label', var_name='Type', value_name='Count')
# melted_df['Actual Label'] = melted_df['Actual Label'].map(lambda x: diseases[x])

# bar_plot = sns.barplot(x='Actual Label', y='Count', hue='Type', data=melted_df, palette='viridis', ax=ax[1])

# for p in bar_plot.patches:
#     bar_plot.annotate(format(p.get_height(), '.1f'), 
#                     (p.get_x() + p.get_width() / 2., p.get_height()), 
#                     ha = 'center', va = 'center', 
#                     xytext = (0, 10), 
#                     textcoords = 'offset points')

# ax[1].set_title('Total Count and Successful Prediction Count for Each Disease')
# ax[1].set_ylabel('Count')
# ax[1].legend(frameon=True, title='Type', title_fontsize='13', loc='upper right')
# ax[1].set_xlabel('Disease')

# plt.tight_layout()
# plt.show()
