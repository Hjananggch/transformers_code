from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load the model
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\AN\Desktop\dj_zsk_chatbot\bge-large-zh-v1.5")
model = AutoModel.from_pretrained(r"C:\Users\AN\Desktop\dj_zsk_chatbot\bge-large-zh-v1.5")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


class Kb:
    def __init__(self,filepath):
        with open(filepath, 'r',encoding='gbk') as f:
            content = f.read()
        self.content = content
        self.docs = self.split_content(256)
        self.ebd = self.encode(self.docs)

    def split_content(self,max_len256):
        chunks = []
        for i in range(0, len(self.content), max_len256):
            chunks.append(self.content[i:i + max_len256])
        return chunks

    @staticmethod
    def encode(doc):
        ebd = []
        for i in doc:
            encoded_input = tokenizer(i, padding=True, truncation=True, return_tensors='pt', max_length=512)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]  # embeddings of [CLS] token
            ebd.append(sentence_embeddings[0].detach().cpu().numpy())
        return ebd

    @staticmethod
    def similarity(A,B):
        sim = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
        return sim

    def search(self,query):
        max_similarity = 0
        max_similarity_index = 0
        query_ebd = self.encode([query])
        for idx,te in enumerate(self.ebd):
            sim = self.similarity(query_ebd[0],te)
            if sim > max_similarity:
                max_similarity = sim
                max_similarity_index = idx
        return self.docs[max_similarity_index]


prompt = "爱因斯坦是什么时候出生的？"
kb = Kb('./data_sample/cndqsjlg.txt')
content = kb.search(prompt)


import requests
url = "http://127.0.0.1:6000/chat"

prompt_pro = f"内容：{content} \n 问题：{prompt} \n请根据内容适当回答问题"
data = {
    "role": "user",
    "content": prompt_pro
}

res = requests.post(url, json=data)
print(res.json())