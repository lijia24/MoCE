import pandas as pd

df1 = pd.read_csv('/mnt/moonfs/lijia-m2/code02/MoTE/lists/ucf101/ucf101_train_idx2cls.csv')          # 第一份不动
df2 = pd.read_csv('/mnt/moonfs/lijia-m2/code02/MoTE/lists/ucf101/ucf101_test_idx2cls.csv')
df2['id'] = df2['id'].astype(int) + 51  # 整体 +51

merged = pd.concat([df1, df2], ignore_index=True)
merged.to_csv('/mnt/moonfs/lijia-m2/code02/MoTE/lists/ucf101/ucf101_label_merged.csv', index=False)