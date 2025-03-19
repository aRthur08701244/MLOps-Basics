import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        input_ids = torch.tensor([processed["input_ids"]]).to(self.device)
        attention_mask = torch.tensor([processed["attention_mask"]]).to(self.device)
        logits = self.model(input_ids, attention_mask)
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=2-step=804.ckpt")
    print(predictor.predict(sentence))
