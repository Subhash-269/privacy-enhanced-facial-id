import pickle

# Replace 'my_file.pkl' with your pickle file path
pickle_file_path = "encrypted_data_.pkl"

with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)

print(data)
