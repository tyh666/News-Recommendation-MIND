
# def news_token_generator(news_file_list, tokenizer, attrs):
#     """ merge and deduplicate training news and testing news then iterate, collect attrs into a single sentence and generate it

#     Args:
#         tokenizer: torchtext.data.utils.tokenizer
#         attrs: list of attrs to be collected and yielded
#     Returns:
#         a generator over attrs in news
#     """
#     news_df_list = []
#     for f in news_file_list:
#         news_df_list.append(pd.read_table(f, index_col=None, names=[
#                             "newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3))

#     news_df = pd.concat(news_df_list).drop_duplicates().dropna()
#     news_iterator = news_df.iterrows()

#     for _, i in news_iterator:
#         content = []
#         for attr in attrs:
#             content.append(i[attr])

#         yield tokenizer(" ".join(content))


# def construct_vocab(news_file_list, attrs):
#     """
#         Build field using torchtext for tokenization

#     Returns:
#         torchtext.vocabulary
#     """
#     tokenizer = get_tokenizer("basic_english")
#     vocab = build_vocab_from_iterator(
#         news_token_generator(news_file_list, tokenizer, attrs))

#     # adjustments for torchtext >= 0.10.0
#     # vocab.insert_token('[PAD]', 0)
#     # vocab.insert_token('[UNK]', 0)
#     # vocab.set_default_index(0)

#     output = open(
#         "data/dictionaries/vocab_{}.pkl".format(",".join(attrs)), "wb")
#     pickle.dump(vocab, output)
#     output.close()


# def construct_basic_dict(attrs=['title','abstract','category','subcategory'], path="../../../Data/MIND"):
#     """
#         construct basic dictionary
#     """
#     news_file_list = [path + "/MINDlarge_train/news.tsv", path +
#                        "/MINDlarge_dev/news.tsv", path + "/MINDlarge_test/news.tsv"]
#     construct_vocab(news_file_list, attrs)

#     for scale in ["demo", "small", "large"]:
#         news_file_list = [path + "/MIND{}_train/news.tsv".format(
#             scale), path + "/MIND{}_dev/news.tsv".format(scale), path + "/MIND{}_test/news.tsv".format(scale)]
#         behavior_file_list = [path + "/MIND{}_train/behaviors.tsv".format(
#             scale), path + "/MIND{}_dev/behaviors.tsv".format(scale), path + "/MIND{}_test/behaviors.tsv".format(scale)]

#         if scale == "large":
#             news_file_train = news_file_list[0]
#             news_file_dev = news_file_list[1]
#             news_file_test = news_file_list[2]

#             construct_nid2idx(news_file_train, scale, "train")
#             construct_nid2idx(news_file_dev, scale, "dev")
#             construct_nid2idx(news_file_test, scale, "test")

#             construct_uid2idx(behavior_file_list, scale)

#         else:
#             news_file_list = news_file_list[0:2]

#             news_file_train = news_file_list[0]
#             news_file_dev = news_file_list[1]

#             construct_nid2idx(news_file_train, scale, "train")
#             construct_nid2idx(news_file_dev, scale, "dev")

#             behavior_file_list = behavior_file_list[0:2]
#             construct_uid2idx(behavior_file_list, scale)


# def construct_vert_onehot():
#     import pandas as pd
#     path = "/home/peitian_zhang/Data/MIND"
#     news_file_list = [path + "/MINDlarge_train/news.tsv", path +
#                         "/MINDlarge_dev/news.tsv", path + "/MINDlarge_test/news.tsv"]
#     news_df_list = []
#     for f in news_file_list:
#         news_df_list.append(pd.read_table(f, index_col=None, names=["newsID", "category", "subcategory", "title", "abstract", "url", "entity_title", "entity_abstract"], quoting=3))

#     news_df = pd.concat(news_df_list).drop_duplicates()

#     vert = news_df["category"].unique()
#     subvert = news_df["subcategory"].unique()
#     vocab = getVocab("data/dictionaries/vocab_whole.pkl")
#     vert2idx = {
#         vocab[v]:i for i,v in enumerate(vert)
#     }
#     subvert2idx = {
#         vocab[v]:i for i,v in enumerate(subvert)
#     }
#     vert2onehot = {}
#     for k,v in vert2idx.items():
#         a = np.zeros((len(vert2idx)))
#         index = np.asarray([v])
#         a[index] = 1
#         vert2onehot[int(k)] = a.tolist()
#     vert2onehot[1] = [0]*len(next(iter(vert2onehot.values())))

#     subvert2onehot = {}
#     for k,v in subvert2idx.items():
#         a = np.zeros((len(subvert2idx)))
#         index = np.asarray([v])
#         a[index] = 1
#         subvert2onehot[int(k)] = a.tolist()
#     subvert2onehot[1] = [0]*len(next(iter(subvert2onehot.values())))

#     json.dump(vert2onehot, open("data/dictionaries/vert2onehot.json","w"),ensure_ascii=False)
#     json.dump(subvert2onehot, open("data/dictionaries/subvert2onehot.json","w"),ensure_ascii=False)