import torch
from transformers import BertJapaneseTokenizer, BertModel
import logging
import numpy as np
logging.basicConfig(level=logging.ERROR)
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')

#関数保管場所
#safe,outファイルを読み込み
def lode_file(filename):
    texts, scores = [], []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            word, score = line.strip().split(',')
            texts.append(word)
            scores.append(score)
    return texts, scores

#ベクトル化処理
def text_vec(text):
    tmp = tokenizer.encode_plus(text, truncation=True, padding=False, return_tensors='pt')
    outputs = model(**tmp)
    return outputs.pooler_output.detach().numpy()[0]

#ファイルのベクトル化
def list_vec(texts_list, scores_list, label):
    vectors, sources = [], []
    for text, score in zip(texts_list, scores_list):
        vectors.append(text_vec(text))
        sources.append((text, label, score))
    return vectors, sources

#コサイン類似度を求める
def comp_sim(qvec,tvec):
    return np.dot(qvec, tvec) / (np.linalg.norm(qvec) * np.linalg.norm(tvec))


#listの平均値算出
def average_file(filename):
    number = []
    with open(filename, 'r', encoding="utf-8") as file:
         for line in file:
             number.append(float(line.strip()))
    return float(sum(number) / len(number)) if len(number) != 0 else 0.75

#safe,outファイルを読み込みベクトル化
text_safe, score_safe = lode_file("safe.txt")
text_out, score_out = lode_file("out.txt")

vec_safe, sources_safe = list_vec(text_safe, score_safe, label="safe")

vec_out, sources_out = list_vec(text_out, score_out, label="out")

vec = vec_safe + vec_out
text_sources = sources_safe + sources_out


#text_xを受け取りベクトル化、類似度を測り、判定を出力
similarity_score = []
defferent_sim = []

text_x = input('判定したいテキストを入力して下さい：')

vec_x = text_vec(text_x)

for tvec in vec:
    similarities = comp_sim(vec_x, tvec)
    similarity_score.append(similarities)
most_similar_index = np.argmax(similarity_score)
most_similar_text, source_file ,B = text_sources[most_similar_index]
most_similar_score = similarity_score[most_similar_index]

F = 210970
F = int(input("フォロワー数："))
if source_file == "safe":
    P = 0
elif source_file == "out":
    P = 1
I = int(F * 0.3 + F ** 0.1 * (1 + 210970 * (int(B) / 100) ** 3.2 * (1 + 0.5 * (int(B) / 100) ** 5 * P)))
R = int(I * 0.01 * (1 + 2 * (int(B) / 100) ** 2) * (1 + P))
L = int(I * 0.03 * (1 + 0.5 * (int(B) / 100) ** 0.7) * (1 + 0.1 * P))
print("似ている文：" + most_similar_text + "、類似度：" + str(most_similar_score))
print("インプレッション数：" + str(I) + "、リポスト数：" + str(R) + "、いいね数：" + str(L))
if most_similar_score < average_file('different_sim.txt'): # 卍要検討卍
  print("判定不可")
else:
  if "safe" in source_file:
      print("判定：SAFE、バズスコア：" + B)
  elif "out" in source_file:
      print("判定：OUT、バズスコア：" + B)

check = str(input('あなたの判定は？(safe/out):'))

if check != source_file:
    with open('different_sim.txt', 'a', encoding="utf-8") as file:
        file.write(str(similarities))