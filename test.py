import os
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from utils.data_utils import load_pretrained_glove_model, preprocess_text_for_html, get_web_files, filter_empty_files, TextWithLabelDataset, collate_fn
from models.FFAN import FFANModel
from models.classifier import Classifier
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()


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


def evaluate_model(texts, labels, file_names, ffan_model, classifier, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextWithLabelDataset(texts, labels, tokenizer, glove_model, file_names)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels, _ in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            features = ffan_model(sequences)
            logits = classifier(features)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":

    folder_paths = [
        r'dataset_demo\JNR3210_V1.1.0.14_1.0.14',
        r'dataset_demo\DIR816L_FW100b09',
        r'dataset_demo\FW_EA6700_1.1.40.176451',
        r'dataset_demo\Archer_c5v2_us-up-ver3-17-1-P1[20150908-rel43260]',
        r'dataset_demo\DIR-300_fw_revb_205b03_ALL_de_20101123',
        r'dataset_demo\R7000_V1.0.4.18_1.1.52',
    ]

    glove_file_path = r'Static_word_embedding_pre-trained_models/glove.6B.300d.txt'
    glove_model = load_pretrained_glove_model(glove_file_path)

    all_texts = []
    all_labels = []
    all_file_names = []
    folder_labels = []

    for path in folder_paths:
        files = get_web_files(path)
        files = filter_empty_files(files)
        texts = [preprocess_text_for_html(open(file, 'r', encoding='iso-8859-1').read()) for file in files]
        all_texts.extend(texts)
        all_file_names.extend(files)
        folder_labels.extend([os.path.basename(path)] * len(files))

    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(folder_labels)

    train_texts, test_texts, train_labels, test_labels, train_file_names, test_file_names = train_test_split(all_texts, all_labels, all_file_names, test_size=0.2, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextWithLabelDataset(train_texts, train_labels, tokenizer, glove_model, train_file_names)
    embedding_matrix = dataset.embedding_matrix

    num_classes = len(set(all_labels))

    trained_ffan_model, trained_classifier = load_model('trained_models/FFAN_model_topK.pth',
                                                        'trained_models/classifier_topK.pth', embedding_matrix, num_classes, device)

    evaluate_model(test_texts, test_labels, test_file_names, trained_ffan_model, trained_classifier, device)
