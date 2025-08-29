import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 1. 读取Excel
df = pd.read_excel(os.path.expanduser('~/Data/AID/all.xlsx'), sheet_name='pre')

# 改成简洁列名
report_col = 'mra_report'
label_col = 'type（1-GC+cs，2-GC+b/ts，3-others（单GC or GC+中药/HCQ，4-上述药物均无）'

texts = df[report_col].astype(str).tolist()
labels = df[label_col].astype(str).tolist()

# 2. 标签编码
le = LabelEncoder()
labels = le.fit_transform(labels)
num_labels = len(le.classes_)

# 3. 切分训练/测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 4. 加载BioClinicalBERT
model_name = os.path.expanduser('~/Data/AID/TAK/Bio_ClinicalBERT')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 5. 创建Dataset类
class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ReportDataset(train_texts, train_labels, tokenizer)
test_dataset = ReportDataset(test_texts, test_labels, tokenizer)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",             # 每个epoch保存模型
    save_total_limit=1,                # 只保留最近一个checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"                   # 不使用wandb
)

# 7. 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 8. 训练模型
trainer.train()

# 9. 模型评估
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
print(classification_report(test_labels, preds, target_names=le.classes_))
