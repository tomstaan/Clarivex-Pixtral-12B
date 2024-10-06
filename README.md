# Clarivex-Pixtral-12B

The **Clarivex-Pixtral-12B** model is a fine-tuned version of the Pixtral-12B model, created during the Hack UK - [a16z](https://a16z.com/) and [Mistral](https://mistral.ai/) AI London Hackathon in October 2024. It has been adapted specifically for the Diverse Dermatology Images (DDI) Multimodal Dataset using Parameter-Efficient Fine-Tuning (PEFT) techniques like Low-Rank Adaptation (LoRA). The fine-tuning was completed on a single Nvidia H100 80GB GPU (Sponsor: [Nebius](https://nebius.ai/)), optimizing the model to handle both visual and clinical data for accurate skin disease diagnosis across diverse populations.


## Diverse Dermatology Images (DDI) Multimodal Dataset [Download](https://www.kaggle.com/datasets/souvikda/ddidiversedermatology-multimodal-dataset/data)

The Diverse Dermatology Images (DDI) Multimodal Dataset is a comprehensive collection of dermatological images paired with clinical metadata, aimed at improving machine learning models for skin disease diagnosis across diverse populations. This dataset emphasizes inclusivity by featuring a wide range of skin tones, and conditions, addressing biases in traditional datasets. The multimodal nature of the dataset, combining both visual and clinical data, makes it a valuable resource for training models that can generalize across skin types, ultimately fostering more equitable and accurate dermatological care.

To use this dataset more effectively with **Mixtral models**, it's recommended to convert all images to `.jpg` format as it is more compatible and efficient for processing. Additionally, to prepare the dataset in a format suitable for Hugging Face, you can utilize the provided `create_hf_dataset.py` script, which formats the dataset for seamless integration with Hugging Face's ecosystem.

## Model

The `Pixtral-12B` is a 12-billion-parameter visual-language model (VLM). It supports applications such as image captioning and visual question answering. More details can be found [here](https://huggingface.co/mistral-community/pixtral-12b).

## Team

The team is listed in alphabetical order:
- Jacob Walker - [LinkedIn](https://www.linkedin.com/in/jnrwalker/)
- Lei Xun - [LinkedIn](https://www.linkedin.com/in/lx2u16/)
- Tomasz Stankiewicz - [LinkedIn](https://www.linkedin.com/in/tomstan/)
