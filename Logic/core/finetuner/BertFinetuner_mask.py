import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score , classification_report

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer,AutoModelForSequenceClassification, get_linear_schedule_with_warmup,TrainingArguments,Trainer


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.dataset = None
        self.threshold = 0.5
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.top_n_genres = top_n_genres
        self.genre_dic = {}
        self.genre_dic_reverse = []
        self.genre_dic_reverse_1 = {}
        self.genre_distribution = {}
        self.file_path = file_path

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as file:
            self.dataset = json.load(file)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        tmp = []
        for movie in self.dataset:
            if movie['genres'] and movie['first_page_summary'] and len(movie['genres']) > 0:
              if movie['first_page_summary'] == '':
                if movie['summaries']:
                    movie['first_page_summary'] = movie['summaries'][0]
                else:
                  continue
              tmp.append(movie)

        self.dataset = tmp
        for movie in self.dataset:
            genres = movie['genres']
            for genre in genres:
                if genre in self.genre_distribution:
                    self.genre_distribution[genre] += 1
                else:
                    self.genre_distribution[genre] = 1
        print(self.genre_distribution)
        self.genre_distribution['Action/Adventure'] = self.genre_distribution.pop(
            'Action') + self.genre_distribution.pop('Adventure')
        self.genre_distribution['Sci-Fi/Fantasy'] = self.genre_distribution.pop('Sci-Fi') + self.genre_distribution.pop(
            'Fantasy')
        self.genre_distribution = dict(
            sorted(self.genre_distribution.items(), key=lambda x: x[1], reverse=True)[:self.top_n_genres])
        print(self.genre_distribution)
        key_genre = list(self.genre_distribution.keys())
        i = 0
        for key in key_genre:
            self.genre_dic[key] = i
            self.genre_dic_reverse_1[i] = key
            self.genre_dic_reverse.append(key)
            i += 1

        self.genre_dic['Action'] = self.genre_dic['Action/Adventure']
        self.genre_dic['Adventure'] = self.genre_dic['Action/Adventure']
        self.genre_dic['Fantasy'] = self.genre_dic['Sci-Fi/Fantasy']
        self.genre_dic['Sci-Fi'] = self.genre_dic['Sci-Fi/Fantasy']

        labels_matrix = np.zeros((len(self.dataset), self.top_n_genres))
        for i, movie in enumerate(self.dataset):
            genres = movie['genres']
            if genres:
                for genre in genres:
                    if genre in self.genre_dic:
                        labels_matrix[i][self.genre_dic[genre]] = 1
            movie['genres'] = labels_matrix[i]



    def split_dataset(self, test_size=0.1, val_size=0.1):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        val_size = (val_size)/(1-test_size)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=test_size, random_state=42)
        self.train_dataset, self.val_dataset = train_test_split(self.train_dataset, test_size=val_size, random_state=42)

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=7, batch_size=128, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        huggingface_model = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(huggingface_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(huggingface_model,
                                                           problem_type="multi_label_classification", id2label=self.genre_dic_reverse_1,label2id = self.genre_dic,

                                                           num_labels=self.top_n_genres)


        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-8, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=len(self.train_dataset) * epochs)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.train()
        train_encodings = self.tokenizer([movie['first_page_summary'] for movie in tqdm(self.train_dataset)],
                                         truncation=True, padding=True)
        val_encodings = self.tokenizer([movie['first_page_summary'] for movie in tqdm(self.val_dataset)],
                                       truncation=True,
                                       padding=True)
        train_labels = [movie['genres'] for movie in self.train_dataset]
        val_labels = [movie['genres'] for movie in self.val_dataset]
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)
        training_arguments = TrainingArguments(
            output_dir=".",
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset= val_dataset,
            optimizers=(optimizer, scheduler),
        )
        trainer.train()



    def compute_metrics(self,eval_pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        scores = {}
        logits, true_labels = eval_pred
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= self.threshold)] = 1
        y_true = true_labels
        report = classification_report(y_true,y_pred, target_names=self.genre_dic_reverse)
        return report


    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_labels = [movie['genres'] for movie in self.test_dataset]
        test_encodings = self.tokenizer([movie['first_page_summary'] for movie in tqdm(self.test_dataset)], truncation=True,
                                   padding=True)
        test_dataset = self.create_dataset(test_encodings, test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        self.model.eval()
        predictions = []
        true_labels = []
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(logits.tolist())
            true_labels.extend(labels.tolist())
        print(self.compute_metrics((predictions,test_labels)))

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained("bert/"+model_name)
        self.tokenizer.save_pretrained("bert/"+model_name)
        self.model.push_to_hub(model_name)
        self.tokenizer.push_to_hub(model_name)





class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
