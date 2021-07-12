import re

from tqdm import tqdm

from image_captioning.constants import TOKENIZER_PATH, PREPROCESSED_TRAIN_DF


def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')


def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')


class Tokenizer(object):
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


def create_tokenizer_and_df(train_df):
    # ====================================================
    # preprocess train.csv
    # ====================================================
    train_df['InChI_1'] = train_df['InChI'].progress_apply(lambda x: x.split('/')[1])
    train_df['InChI_text'] = (
        train_df['InChI_1'].progress_apply(split_form)
        + ' '
        + train_df['InChI']
        .apply(lambda x: '/'.join(x.split('/')[2:]))
        .progress_apply(split_form2)
        .values
    )
    # ====================================================
    # create tokenizer
    # ====================================================
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['InChI_text'].values)
    torch.save(tokenizer, TOKENIZER_PATH)
    print(f'Saved tokenizer: {TOKENIZER_PATH}')
    # ====================================================
    # preprocess train.csv
    # ====================================================
    lengths = []
    tk0 = tqdm(train_df['InChI_text'].values, total=len(train_df))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    train_df['InChI_length'] = lengths
    train_df.to_pickle(PREPROCESSED_TRAIN_DF)
    print(f'Saved preprocessed {PREPROCESSED_TRAIN_DF}')


def apply_tokenizer_to_df(tokenizer, train_df):
    # ====================================================
    # preprocess train.csv
    # ====================================================
    train_df['InChI_1'] = train_df['InChI'].progress_apply(lambda x: x.split('/')[1])
    train_df['InChI_text'] = (
        train_df['InChI_1'].progress_apply(split_form)
        + ' '
        + train_df['InChI']
        .apply(lambda x: '/'.join(x.split('/')[2:]))
        .progress_apply(split_form2)
        .values
    )
    # ====================================================
    # preprocess train.csv
    # ====================================================
    lengths = []
    tk0 = tqdm(train_df['InChI_text'].values, total=len(train_df))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    train_df['InChI_length'] = lengths
    train_df.to_pickle(PREPROCESSED_TRAIN_DF)
    print(f'Saved preprocessed {PREPROCESSED_TRAIN_DF}')