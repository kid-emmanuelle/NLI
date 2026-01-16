import pandas as pd

def load_data_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

df_test_matched = load_data_jsonl('data/multinli_1.0_dev_matched.jsonl')
df_test_mismatched = load_data_jsonl('data/multinli_1.0_dev_mismatched.jsonl')
df_train = load_data_jsonl('data/multinli_1.0_train.jsonl')

