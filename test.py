import numpy as np

dataset = 'swat'
train_data_file =  "data/synth/X_train.npy" #"data/swat/SWaT_minute_segments_normal.npy"#
train_data = np.load(train_data_file)

print(f"train data: {train_data.shape}")
train_data = train_data.reshape((-1, train_data.shape[-1]))
mean = np.nanmean(train_data, axis=0)
std = np.nanstd(train_data, axis=0)
train_data = (train_data - mean) / (std + 1)
train_data = np.nan_to_num(train_data, copy=True)
print(f"train data: {train_data.shape}")
np.save(f"data/{dataset}/mean.npy", mean)
np.save(f"data/{dataset}/std.npy", std)
np.save(train_data_file, train_data)

test_data_file = "data/synth/X_test.npy" #"data/swat/SWaT_minute_segments_anomaly.npy" #
test_data = np.load(test_data_file)

print(f"test data: {test_data.shape}")

test_data = test_data.reshape((-1, test_data.shape[-1]))
test_data = (test_data - mean) / (std + 1)
test_data = np.nan_to_num(test_data, copy=True)
print(f"test data: {test_data.shape}")
np.save(test_data_file, test_data)

label_data_file = "data/synth/Y_test.npy" #"data/swat/SWaT_minute_segments_anomaly_labels.npy"#
label_data = np.load(label_data_file)

print(f"label data: {label_data.shape}")
# label_data = label_data.reshape(-1)
label_data = label_data.reshape((-1, label_data.shape[-1]))
label_data = np.max(label_data, axis=-1)
print(f"label data: {label_data.shape}")
np.save(label_data_file, label_data)
