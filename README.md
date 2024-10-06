# clarivex-model

## [Diverse Dermatology Images (DDI) Multimodal Dataset](https://www.kaggle.com/datasets/souvikda/ddidiversedermatology-multimodal-dataset/data)

The Diverse Dermatology Images (DDI) Multimodal Dataset is a comprehensive collection of dermatological images paired with clinical metadata, aimed at improving machine learning models for skin disease diagnosis across diverse populations. This dataset emphasizes inclusivity by featuring a wide range of skin tones, and conditions, addressing biases in traditional datasets. The multimodal nature of the dataset, combining both visual and clinical data, makes it a valuable resource for training models that can generalize across skin types, ultimately fostering more equitable and accurate dermatological care.

To use this dataset more effectively with **Mixtral models**, it's recommended to convert all images to `.jpg` format as it is more compatible and efficient for processing. Additionally, to prepare the dataset in a format suitable for Hugging Face, you can utilize the provided `create_hf_dataset.py` script, which formats the dataset for seamless integration with Hugging Face's ecosystem.