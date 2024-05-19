# Пример использования трансформера из pytorch
# разработан на основе
# https://github.com/pytorch/examples/blob/main/word_language_model/main.py
# Также можно использовать пример
# https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/demos/transformer.py

# После запуска программа подгружает текст выборки Wikitext-2 и пытается обучиться на нём генерации схожих
# текстов. При обучении лучшие из имеющихся сетей сохраняются в файле "model.pt". Если хочется использовать
# сеть из этого файла, не дожидаясь повторного обучения, можно нажать Ctrl-C, и программа перейдёт к 
# генерации текста.

# При начальных прогонах рекомендуется использовать малое число эпох обучения и мало данных,
# чтобы убедиться в работоспособности программы; для этого следует снизить параметр NUM_EPOCH
# и вручную укоротить файл data/wikitext-2.txt с обучающими данными. По мере настройки можно вернуть начальные
# значения.

# При генерации полезно изучить влияние параметра "температуры" на результаты.

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

''' класс словаря. Будет содержать список всех слов и их номеров'''
class Dictionary(object):
    def __init__(self):
        self.idx2word = [] # список всех слов для доступа к слову по номеру
        self.word2idx = {} # словарь для доступа к номеру по слову
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

''' класс корпуса текстов. Содержит список слов Dictionary
и обучающую выборку. Желательно добавить в него тестовую и валидационную выборки
для проверки качества работы '''
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path)

    def tokenize(self, path):
        # занести файл в словарь;
        # заменить слова в содержащемся в нём тексте номерами,
        # выдать список этих номеров
        assert os.path.exists(path)
        # добавить слова в словарь
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>'] # отдельный символ конца строки
                for word in words:
                    self.dictionary.add_word(word)

        # заменить слова номерами
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class PositionalEncoding(nn.Module):
    # класс для внесения информации о позициях слов в токены.
    # Слово без учёта позиции задаётся унитарным кодом --- вектором
    # нулей, среди которых стоит одна единица, причём её место равно номеру слова
    # в словаре. Класс PositionalEncoding добавляет в этот вектор информацию 
    # о позиции слова в последовательности (предложении или строке)
    # способом, описанным в лекции: если позиция слова в последовательность равна pos,
    # то к элементам унарного вектора слова W добавляются значения:
    # W[2i]   = sin(pos/10000^(2i/d_model))
    # W[2i+1] = cos(pos/10000^(2i/d_model))
    # Это позволяет варьировать векторы слова в зависимости от места,
    # на котором оно стоит, чтобы сеть могла учесть место.

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # dropout, как и во всех сетях, помогает бороться с переобучением
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # прямая обработка данных (добавление позиционного кода)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Transformer):
    # Модель трансформера: кодер, трансформер, декодер
    
    # Смысл константных параметров объяснён ниже --- там, где
    # они объявлены
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

# Константы для описания поведения системы
SEED = 21  # начальное значение генератора случайных чисел.
           # задаётся для воспроизведения результатов
BATCH_SIZE=20  # размер куска данных, обрабатываемого за один раз
EMBEDDING_SIZE=200  # размер вектора описания слова;
                    # чем он больше, тем больше информации о слове можно сохранить
                    # и тем выше вероятность переобучения
NUM_HEADS=2  # количество головок самовнимания в трансформере
NUM_HIDDEN=200  # число скрытых нейронов на один слой трансформера
NUM_LAYERS=2  # число слоёв трансформера
DROPOUT=0.2   # доля выбрасываемых при обучении данных
SEQ_LEN=35    # длина последовательности, на которые разбивается текст при обучении
GRADIENT_CLIPPING = 0.25  # коэффициент снижения градиента для стабилизации обучения
INIT_LEARNING_RATE = 20.  # первоначальная скорость обучения; в процессе работы снижается
NUM_EPOCH = 10  # число эпох обучения
INP_FILENAME = './data/wikitext-2.txt'  # файл с данными
OUT_FILENAME = 'model.pt'  # файл для сохранения модели

# Установим генератор случайных чисел
torch.manual_seed(SEED)
# Укажем выполнять обработку на процессоре
device = torch.device("cpu")

def download(destination):
    # функция загрузки данных
    if os.path.exists(destination):
        return
    import requests
    #os.makedirs(destination.parent, exist_ok=True)
    url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    with open(destination, "w") as f:
        f.write(requests.get(url).text)

# если файла с корпусом текстов нет на месте, загрузим его
if not os.path.exists(INP_FILENAME):
    download(INP_FILENAME)

# Загрузим корпус текстов из файла
corpus = Corpus(INP_FILENAME)

# Разобьём последовательность на неперекрывающиеся куски. Так, последовательность
# abcdefgh... при разбиении на куски по 6 элементов образует колонки матрицы:
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# Каждая колонка считается при обучении независимой последовательностью,
# т.е. f и g, находящиеся в оригинальном тексте рядом, не повлияют друг на друга.
def batchify(data, bsz):
    # data --- данные, bsz --- размер одного куска
    nbatch = data.size(0) // bsz
    # хвост отбрасываем
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, BATCH_SIZE)

# Создаём модель и критерий минимизации
ntokens = len(corpus.dictionary)
model = TransformerModel(ntokens, EMBEDDING_SIZE, NUM_HEADS,
                         NUM_HIDDEN, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.NLLLoss()

# Дополнительно разобъём данные на куски длины SEQ_LEN для поочерёдной обработки
def get_batch(source, i):
    seq_len = min(SEQ_LEN, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    # функция для проверки работы сети
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # функция обучения сети
    logging_interval = 100
    model.train()
    sum_loss = 0.
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, SEQ_LEN)):
        data, targets = get_batch(train_data, i)
        model.zero_grad()
        # подаём данные на сеть, проверяем совпадение, корректируем градиенты
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` уменьшает градиент, чтобы стабилизировать обучение;
        # для трансформера можно его убрать
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        
        # выводим текущие результаты. Здесь правильно было бы
        # добавить валидацию, чтобы оценивать результаты не на обучающей выборке
        if batch % logging_interval == 0 and batch > 0:
            cur_loss = total_loss / logging_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // SEQ_LEN, lr,
                elapsed * 1000 / logging_interval, cur_loss, math.exp(cur_loss)))
            sum_loss += total_loss
            total_loss = 0
            start_time = time.time()
    sum_loss += total_loss
    return sum_loss

# Большой цикл обучения
lr = INIT_LEARNING_RATE  # текущая скорость обучения, которая потом снижается.
best_loss = None  # лучший из достигнутых результатов

# Чтобы прервать обучение, можно нажать Ctrl + C и перейти сразу к генерации текста
try:
    for epoch in range(1, NUM_EPOCH+1):
        epoch_start_time = time.time()
        tr_loss = train()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.2f}'
              .format(epoch, (time.time() - epoch_start_time),
                      tr_loss))
        print('-' * 89)
        # сохранить лучшую модель
        if not best_loss or tr_loss < best_loss:
            with open(OUT_FILENAME, 'wb') as f:
                torch.save(model, f)
            best_loss = tr_loss
        else:
            # попытаться снизить скорость обучения, если старая не позволяет улучшить результаты
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Загрузить лучшую из моделей
with open(OUT_FILENAME, 'rb') as f:
    model = torch.load(f)

# Генерация текста
NUM_WORDS = 100  # требуемая длина текста
TEMPERATURE = 1.35 # показатель хаотичности текста. При слишком высокой температуре
                 # текст слишком хаотичный (до потери связности), при слишко низкой
                 # копирует обучающую выборку или зацикливается на одном слове
test_input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
words = []
word_ids = []

model.eval()
with torch.no_grad():
    for temperature in (0,9, 1, 1.5, 3, 5):
        for i in range(NUM_WORDS):
            output = model(test_input, False)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            word = corpus.dictionary.idx2word[word_idx]
            test_input = torch.cat([test_input, word_tensor], 0)
            words.append(word)
            word_ids.append(word_idx)

print(' '.join(words))
exit()