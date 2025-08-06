import torch
import json
import os
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils.data_utils import load_pretrained_glove_model, preprocess_text_for_html, get_web_files, filter_empty_files, TextWithLabelDataset, collate_fn
from models.FFAN import FFANModel
from models.classifier import Classifier


def load_model(ffan_model_path, classifier_path, embedding_matrix, num_classes, device):
    ffan_model = FFANModel(embedding_matrix)
    ffan_model.load_state_dict(torch.load(ffan_model_path, map_location=device))
    ffan_model.to(device)
    ffan_model.eval()

    classifier = Classifier(input_dim=ffan_model.hidden_dim * 3, num_classes=num_classes)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    return ffan_model, classifier


def generate_embeddings_for_new_firmware(firmware_folders, ffan_model, classifier, device, glove_model, tokenizer):
    embeddings_output = {}

    # Loop through the new firmware folders
    for folder in firmware_folders:
        files = get_web_files(folder)
        files = filter_empty_files(files)

        for file in files:
            # Read and preprocess the HTML content
            with open(file, 'r', encoding='iso-8859-1') as f:
                text = f.read()
            text = preprocess_text_for_html(text)

            # Create the dataset for this specific file
            dataset = TextWithLabelDataset([text], [0], tokenizer, glove_model, [file])
            dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

            with torch.no_grad():
                for sequences, _, file_names in dataloader:
                    sequences = sequences.to(device)
                    features = ffan_model(sequences)  # Generate features using FFAN model
                    embeddings_output[f"{os.path.basename(folder)}|||{file}"] = features.cpu().numpy().tolist()

    return embeddings_output


def save_embeddings_to_json(embeddings, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(embeddings, json_file, indent=4)


if __name__ == "__main__":
    root_dir = r'D:\DeepFW\test-dataset'
    firmware_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d))]
    # List of new firmware version directories to test

    print(firmware_folders)
    # firmware_folders = [
    #     r'dataset_demo/New_Firmware_1',
    #     r'dataset_demo/New_Firmware_2',
    #     r'dataset_demo/New_Firmware_3',
    #     r'dataset_demo/New_Firmware_4'
    # ]

    glove_file_path = r'Static_word_embedding_pre-trained_models/glove.6B.300d.txt'
    glove_model = load_pretrained_glove_model(glove_file_path)

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load embedding matrix for the GloVe model
    dataset = TextWithLabelDataset([], [], tokenizer, glove_model, [])
    embedding_matrix = dataset.embedding_matrix

    num_classes = 6  # Modify this to match the number of classes in the trained model (e.g., 6)

    trained_ffan_model, trained_classifier = load_model(
        'trained_models/FFAN_model_topK.pth',
        'trained_models/classifier_topK.pth',
        embedding_matrix,
        num_classes,
        device
    )

    # Generate embeddings for the new firmware versions
    embeddings = generate_embeddings_for_new_firmware(firmware_folders, trained_ffan_model, trained_classifier, device,
                                                      glove_model, tokenizer)

    # Save the embeddings to a JSON file
    output_json_path = './FirmID-predictions/firmware_web_embeddings.json'
    save_embeddings_to_json(embeddings, output_json_path)

    print(f"Embeddings for new firmware have been saved to: {output_json_path}")
