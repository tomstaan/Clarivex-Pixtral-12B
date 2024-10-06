import os
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image

# Define the path to your images
data_dir = "./data_raw"

# Define the paths to your train and test CSV files
train_csv_file = os.path.join(data_dir, "training.csv")
test_csv_file = os.path.join(data_dir, "test.csv")


def load_image(image_path):
    with open(image_path, 'rb') as image_file:
        return image_file.read()

def create_dataset(csv_file, split_name):
    df = pd.read_csv(csv_file)

    # Filter out rows with empty captions (if any)
    df = df.dropna(subset=['caption'])

    # Load images and update the 'image' column
    df['image'] = df['image_name'].map(lambda image_name: load_image(os.path.join(data_dir, image_name)))

    # Remove the 'image_name' column as it's no longer needed
    df = df.drop(columns=['image_name'])

    # Reset the index to avoid '__index_level_0__' column
    df = df.reset_index(drop=True)

    # Define the features of the dataset
    features = Features({
        'image': Image(),
        'caption': Value('string')
    })

    # Create a Dataset object from the DataFrame
    dataset = Dataset.from_pandas(df, features=features)

    return dataset


# Create train and test datasets
train_dataset = create_dataset(train_csv_file, "train")
test_dataset = create_dataset(test_csv_file, "test")

# Combine train and test into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

save_path = "./data"
dataset_dict.save_to_disk(save_path)

