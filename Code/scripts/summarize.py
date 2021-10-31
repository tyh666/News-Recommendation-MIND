from keybert import KeyBERT
from tqdm import tqdm

kw_model = KeyBERT()

news_file_path_list = ["/data/workspace/Peitian/Data/MIND/MINDdemo_dev/", "/data/workspace/Peitian/Data/MIND/MINDdemo_train/", "/data/workspace/Peitian/Data/MIND/MINDdemo_test/"]

for news_file in news_file_path_list:
    with open(news_file + "news.tsv", "r", encoding="utf-8") as rd:
        f = open(news_file + "keywords.tsv", "w", encoding="utf-8")

        for idx in tqdm(rd, ncols=120, leave=True):
            nid, vert, subvert, title, ab, url, title_entity, abs_entity = idx.strip("\n").split("\t")

            article = " ".join([title, ab, subvert])

            kwds = kw_model.extract_keywords(article,keyphrase_ngram_range=(1, 1), stop_words='english')
            keyword = [kwd[0] for kwd in kwds[:10]]

            f.write(' '.join(keyword) + '\n')

        f.close()