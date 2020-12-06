import numpy as np

def read_golden_id_file(path: str):
    data = []
    with open(path, 'r', encoding='UTF-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line:
                data.append(line.split('\t'))
            else:
                data.append([])
    return data

def read_text_file(path: str):
    data = []
    cache_text = []
    cache_label = []
    with open(path, 'r', encoding='UTF-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line:
                word, *labels = line.split('\t')
                label = labels[-1]
                cache_text.append(word)
                cache_label.append(label)
            else:
                data.append([cache_text.copy(), cache_label.copy()])
                cache_text.clear()
                cache_label.clear()
    return data

def map_id_to_text(data_text, data_id):
    assert len(data_text) == len(data_id)

    text_id_data = []
    for text, ids in zip(data_text, data_id):
        label = text[1]
        cache_ids = []
        cnt = 0
        if ids:
            for l in label:
                if l == 'O':
                    if len(cache_ids) and cache_ids[-1] != 'O':
                        cnt += 1
                    cache_ids.append('O')
                elif 'B-' in l:
                    cache_ids.append(ids[cnt])
                elif 'I-' in l:
                    cache_ids.append(ids[cnt])
        else:
            for l in label:
                cache_ids.append('O')
        text.append(cache_ids)
        text_id_data.append(text)
    
    return text_id_data

def split_data(data, size):
    idx = [i for i in range(len(data))]
    selected_ids = np.random.choice(idx, size=size)

    train_data = []
    dev_data = []

    for i in idx:
        if i in selected_ids:
            dev_data.append(data[i])
        else:
            train_data.append(data[i])
    
    return train_data, dev_data

def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        for ele in data:
            for w, l, i in zip(*ele):
                fp.write(f"{w}\t{l}\t{i}\n")
            fp.write('\n')

train_golden_id = read_golden_id_file('train_goldenID.txt')
train_text = read_text_file('train.final.txt')
train_data = map_id_to_text(train_text, train_golden_id)

test_golden_id = read_golden_id_file('test_goldenID.txt')
test_text = read_text_file('test.final.txt')
test_data = map_id_to_text(test_text, test_golden_id)

train_data, dev_data = split_data(train_data, len(test_data))

save_data(train_data,'train.txt')
save_data(dev_data, 'dev.txt')
save_data(test_data, 'test.txt')