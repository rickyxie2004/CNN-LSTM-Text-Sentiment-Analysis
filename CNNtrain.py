import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

train_file = './train.txt'
test_file = './test.txt'
vector_file = './wiki_word2vec_50.bin'
train_words = []
train_result = []
test_words = []
test_result = []

def load_word2vec_bin(file):
    with open(file, 'rb') as f:
        # 读取单词表的大小和向量维度
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())

        # 初始化单词表和向量
        word_vector = {}

        # 读取单词和向量
        binary_len = np.dtype('float32').itemsize * vector_size
        for _ in range(vocab_size):
            # 读取单词
            word = b''
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    word += ch
            word = word.decode('utf-8')

            # 读取向量
            vector = np.frombuffer(f.read(binary_len), dtype='float32')

            # 添加到字典
            word_vector[word] = vector

        return word_vector

# 加载模型
word_vectors = load_word2vec_bin('wiki_word2vec_50.bin')

def load_data(path):
    words = []
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                flag, sentence = int(parts[0]), parts[1]
                words.append(sentence.split())
                result.append(flag)
    return words, result

train_words, train_result = load_data(train_file)
test_words, test_result = load_data(test_file)

# 定义文本转换为向量的函数
def text_to_vector(text):
    if text in word_vectors:
        return word_vectors[text]
    else:
        return [0]*50

# 处理训练集的文本向量
train_vector = []
max_length_train = 37
for line in train_words:
    temp = []
    for i in range(min(len(line), max_length_train)):
        temp.append(text_to_vector(line[i]))
    # 截取或补充0向量
    if len(temp) < max_length_train:
        temp += [[0] * 50] * (max_length_train - len(temp))
    train_vector.append(temp)

# 将输入数据reshape成四维
train_input = np.array(train_vector, dtype='float32')
train_tensor = torch.Tensor(train_input).unsqueeze(1)

# 处理测试集的文本向量
test_vector = []
max_length_test = 37
for line in test_words:
    temp = []
    for i in range(min(len(line), max_length_test)):
        temp.append(text_to_vector(line[i]))
    # 截取或补充0向量
    if len(temp) < max_length_test:
        temp += [[0] * 50] * (max_length_test - len(temp))
    test_vector.append(temp)

# 将输入数据reshape成四维
test_input = np.array(test_vector, dtype='float32')
test_tensor = torch.Tensor(test_input).unsqueeze(1)

# 构建数据集和数据加载器
train_dataset = TensorDataset(train_tensor, torch.LongTensor(train_result))
test_dataset = TensorDataset(test_tensor, torch.LongTensor(test_result))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.5):
        super(CNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout_rate=dropout_rate
        self.convol1 = nn.Sequential(
            nn.Conv2d(1, hidden_size, (7, 50)),
            nn.MaxPool2d((input_size - 6, 1), input_size-6),
            nn.Dropout(p=self.dropout_rate),
            nn.Flatten()
        )
        self.convol2 = nn.Sequential(
            nn.Conv2d(1, hidden_size, (5, 50)),
            nn.MaxPool2d((input_size-4, 1), input_size-4),
            nn.Dropout(p=self.dropout_rate),
            nn.Flatten()
        )
        self.convol3 = nn.Sequential(
            nn.Conv2d(1, hidden_size, (3, 50)),
            nn.MaxPool2d((input_size-2, 1), input_size-2),
            nn.Dropout(p=self.dropout_rate),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*3, output_size),
        )

    def forward(self,x):
        y1=self.convol1(x)
        y2=self.convol2(x)
        y3=self.convol3(x)
        output_bfc=torch.cat((y1,y2,y3),dim=-1)
        output=self.fc(output_bfc)
        return output

# 创建模型、损失函数和优化器，并放到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(37, 50, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# 训练模型
epoch_tot = 10
inspect_inv = 100
min_loss = float(10)

# 用于存储loss和epoch的列表
losses = []
epochs = []

# 创建图形窗口
plt.figure()

for epoch in range(epoch_tot):
    model.train()
    running_loss = 0.0
    for batch, (input_data, expect) in enumerate(train_loader):
        input_data, expect = input_data.to(device), expect.to(device)
        output = model(input_data)
        loss = criterion(output, expect)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch % inspect_inv == 0:
            print('Epoch:', epoch+1, 'Batch:', batch, 'Loss:', loss.item())
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    epochs.append(epoch)

    plt.clf()
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.ylim(0.1, 0.65)  # 设置y轴范围
    plt.yticks(np.arange(0,0.65, 0.05))  # 设置y轴刻度
    plt.pause(0.01)

    if epoch_loss < min_loss:
        min_loss = epoch_loss
        torch.save(model.state_dict(), 'CNNmodel.pt')  # 只保存模型参数

# 测试模型
p=0
n=0
tp=0
fp=0
tn=0
fn=0
with torch.no_grad():
    model.eval()
    correct = 0
    tot = 0
    for batch, (input_data, expect) in enumerate(test_loader):
        input_data, expect = input_data.to(device), expect.to(device)
        output = model(input_data)
        _, predict = torch.max(output, dim=1)
        tot += expect.size(0)
        correct += torch.eq(predict, expect).sum().item()
        p+=torch.sum(expect==0).item()
        n+=torch.sum(expect==1).item()
        tp+=torch.sum((expect==0)&(predict==0)).item()
        fp+=torch.sum((expect==1)&(predict==0)).item()
        tn+=torch.sum((expect==1)&(predict==1)).item()
        fn+=torch.sum((expect==0)&(predict==1)).item()
    accuracy = correct / tot
    precision = tp / (tp + fp)
    recall = tp / p
    f_m = 2 * precision * recall / (precision + recall)
    print('Accuracy:', accuracy)
    print('F-measure:', f_m)


plt.show()  # 显示图形
