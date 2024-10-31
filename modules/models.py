import os
import torch
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CategorizerModel:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        self.model.eval()
        self.model.to(device)

        if "cuda" in str(device):
            self.model.half()

        self.max_length = self.model.config.max_position_embeddings
    
    def predict(self, text):
        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        
        # do the softmax to get the probabilities and send back the id2label-ed string plus the probability in a dict
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy().tolist()
        label_id = torch.argmax(logits, dim=-1).item()
        label = self.model.config.id2label[label_id]
        return {"category": label, "probability": probs[label_id]}

class PriorityModel(CategorizerModel):
    def __init__(self, model_path):
        priority_path = os.path.join(model_path, "priority")
        super().__init__(priority_path)

    def predict(self, text):
        prediction = super().predict(text)
        prediction["category"] = int(prediction["category"])
        return prediction
    
class TypeModel(CategorizerModel):
    def __init__(self, model_path):
        type_path = os.path.join(model_path, "type")
        super().__init__(type_path)

class UserGradeModel(CategorizerModel):
    def __init__(self, model_path):
        user_grade_path = os.path.join(model_path, "user_grade")
        super().__init__(user_grade_path)  

class OslModel(CategorizerModel):
    def __init__(self, model_path):
        osl_path = os.path.join(model_path, "osl")
        super().__init__(osl_path)
        self.load_osl_graph(model_path)
        
    def load_osl_graph(self, model_path):
        with open(os.path.join(model_path, "osl_graph.json"), "r", encoding="utf-8") as f:
            self.graph = json.load(f)
    
    def get_valid_classes(self, text, partner_name=None):
        if partner_name is None:
            partner_name = text.split(" | ")[0]
            partner_name = partner_name.split("Partner: ")[1]

        if partner_name not in self.graph:
            print(f"Partner {partner_name} not found in the operation_service_level graph, returning zero OSL classes")
            return []
        
        return self.graph[partner_name].keys()
    
    def get_valid_class_ids(self, valid_classes):
        ids = []
        for cls in valid_classes:
            if cls in self.model.config.label2id:
                ids.append(self.model.config.label2id[cls])
        return ids

    def predict(self, text, partner_name=None):
        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits

        valid_classes = self.get_valid_classes(text, partner_name)
        if not valid_classes:
            return {"category": None, "probability": 0.0}
        
        # Before we do the softmax, we will filter out the invalid classes
        valid_classes_ids = self.get_valid_class_ids(valid_classes)
        logits = logits[0][valid_classes_ids] # only include the valid classes

        # do the softmax to get the probabilities and send back the id2label-ed string plus the probability in a dict
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy().tolist()
        label_id = torch.argmax(logits, dim=-1).item()
        label = self.model.config.id2label[valid_classes_ids[label_id]]

        return {"category": label, "probability": probs[label_id]}

    
class OsltModel(CategorizerModel):
    def __init__(self, model_path):
        oslt_path = os.path.join(model_path, "oslt")
        super().__init__(oslt_path)
        self.load_oslt_graph(model_path)
        
    def load_oslt_graph(self, model_path):
        with open(os.path.join(model_path, "osl_graph.json"), "r", encoding="utf-8") as f:
            self.graph = json.load(f)

    def get_valid_classes(self, text, partner_name=None):
        if partner_name is None:
            partner_name = text.split(" | ")[0]
            partner_name = partner_name.split("Partner: ")[1]

        if partner_name not in self.graph:
            print(f"Partner {partner_name} not found in the operation_service_level graph, returning all OSLT classes")
            return self.model.config.id2label.values()
        
        valid_osl_classes = self.graph[partner_name].keys()
        valid_oslt_classes = []

        for osl_class in valid_osl_classes:
            valid_oslt_classes.append(self.graph[partner_name][osl_class])
        
        return valid_oslt_classes
    
    def get_valid_class_ids(self, valid_classes):
        ids = []
        for cls in valid_classes:
            if cls in self.model.config.label2id:
                ids.append(self.model.config.label2id[cls])
        return ids
        
    def predict(self, text, partner_name=None):
        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits

        valid_classes = self.get_valid_classes(text, partner_name)
        
        # Before we do the softmax, we will filter out the invalid classes
        valid_classes_ids = self.get_valid_class_ids(valid_classes)
        logits = logits[0][valid_classes_ids]

        # do the softmax to get the probabilities and send back the id2label-ed string plus the probability in a dict
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy().tolist()
        label_id = torch.argmax(logits, dim=-1).item()
        label = self.model.config.id2label[valid_classes_ids[label_id]]

        return {"category": label, "probability": probs[label_id]}