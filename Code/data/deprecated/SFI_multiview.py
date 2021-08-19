# not refactored yet
# class SFI_MultiView(nn.Module):
#     def __init__(self, config, encoder, ranker):
#         super().__init__()

#         self.signal_length = config['title_length'] + 2
#         self.abs_size = config['abs_size']

#         self.k = config['k']

#         if(encoder.name != 'fim' or ranker.name != 'fim'):
#             logging.error("please use FIM encoder and FIM ranker")

#         self.encoder = encoder
#         self.level = encoder.level
#         self.hidden_dim = encoder.hidden_dim

#         self.ranker = ranker
#         if self.k > 9:
#             title_dim = int(int(self.k / 3) /3) * int(int(self.signal_length / 3) / 3)**2 * 16
#             abs_dim = int(int(self.k / 3) /3) * int(int(self.abs_size / 3) / 3)**2 * 16
#         else:
#             title_dim = (self.k - 4) * int(int(self.signal_length / 3) / 3)**2 * 16
#             abs_dim = (self.k - 4)* int(int(self.abs_size / 3) / 3)**2 * 16

#         # final_dim += self.his_size

#         self.view_dim = 200
#         self.title2view = nn.Linear(title_dim,200)
#         self.abs2view = nn.Linear(abs_dim,200)
#         self.viewQuery = nn.Parameter(torch.randn(1,self.view_dim))

#         self.learningToRank = nn.Sequential(
#             nn.Linear(self.view_dim, 50),
#             nn.ReLU(),
#             nn.Linear(50,1)
#         )

#         self.selectionProject = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim)
#         )

#         self.name = '-'.join(['sfi-multiview', encoder.name, ranker.name])

#         if config.threshold != -float('inf'):
#             threshold = torch.tensor([config.threshold])
#             self.register_buffer('threshold', threshold)

#         for param in self.selectionProject:
#             if isinstance(param, nn.Linear):
#                 nn.init.xavier_normal_(param.weight)
#         for param in self.learningToRank:
#             if isinstance(param, nn.Linear):
#                 nn.init.xavier_normal_(param.weight)

#         nn.init.xavier_normal_(self.title2view.weight)
#         nn.init.xavier_normal_(self.abs2view.weight)


#     def hisSelector(self, cdd_repr, his_repr, his_embedding):
#         """ apply news-level attention

#         Args:
#             cdd_repr: tensor of [batch_size, cdd_size, hidden_dim]
#             his_repr: tensor of [batch_size, his_size, hidden_dim]
#             his_embedding: tensor of [batch_size, his_size, signal_length, level, hidden_dim]

#         Returns:
#             his_activated: tensor of [batch_size, cdd_size, k, signal_length, level, hidden_dim]
#             his_focus: tensor of [batch_size, cdd_size, k, his_size]
#             pos_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
#             neg_repr: tensor of [batch_size, cdd_size, contra_num, hidden_dim]
#         """

#         # [bs, cs, hs]
#         # t1 = time.time()
#         cdd_repr = F.normalize(self.selectionProject(cdd_repr),dim=-1)
#         his_repr = F.normalize(self.selectionProject(his_repr),dim=-1)
#         signal_length = his_embedding.size(-3)
#         attn_weights = cdd_repr.matmul(his_repr.transpose(-1, -2))

#         if self.k == self.his_size:

#             his_activated = his_embedding.unsqueeze(dim=1)
#             attn_weights_index = None

#         else:
#             # t2 = time.time()
#             attn_weights, attn_weights_index = attn_weights.topk(dim=-1, k=self.k)

#             # print(attn_weights, attn_weights_index)
#             # t3 = time.time()

#             # [bs, cs, k, sl, level, fn]
#             his_activated = his_embedding.unsqueeze(dim=1).expand(self.batch_size, self.cdd_size, self.his_size, signal_length, self.level, self.hidden_dim).gather(dim=2, index=attn_weights_index.view(self.batch_size,self.cdd_size,self.k,1,1,1).expand(self.batch_size,self.cdd_size,self.k,signal_length,self.level,self.hidden_dim))

#             # t4 = time.time()

#         if hasattr(self,'threshold'):
#             his_activated = his_activated * (attn_weights.masked_fill(attn_weights<self.threshold, 0).view(self.batch_size, self.cdd_size, self.k, 1, 1, 1))

#         # t6 = time.time()
#         # print("product time:{}, sort time:{}, scatter time:{}, activate time:{}, mask time:{}".format(t2-t1, t3-t2, t4-t3, t5-t4, t6-t5))

#         output = (his_activated, attn_weights_index)
#         return output

#     def _click_predictor(self, fusion_tensors):
#         """ calculate batch of click probabolity

#         Args:
#             fusion_tensors: tensor of [batch_size, cdd_size, *]

#         Returns:
#             score: tensor of [batch_size, cdd_size], which is normalized click probabilty
#         """
#         score = self.learningToRank(fusion_tensors).squeeze(dim=-1)
#         return score

#     def _forward(self, x):
#         if x['cdd_encoded_index'].shape[0] != self.batch_size:
#             self.batch_size = x['cdd_encoded_index'].shape[0]

#         # FIXME, according to FIM, the category is concatenated into title before padding
#         cdd_title = torch.cat([x['cdd_encoded_index'], x['candidate_vert'], x['candidate_subvert']], dim=-1).long().to(self.device)
#         cdd_title_embedding, cdd_title_repr = self.encoder(
#             cdd_title,
#             user_index=x['user_index'].long().to(self.device),
#             news_id=x['cdd_id'].long().to(self.device))
#             # attn_mask=x['cdd_encoded_index_pad'].to(self.device))

#         his_title = torch.cat([x["his_encoded_index"], x['clicked_vert'], x['clicked_subvert']], dim=-1).long().to(self.device)
#         his_title_embedding, his_title_repr = self.encoder(
#             his_title,
#             user_index=x['user_index'].long().to(self.device),
#             news_id=x['his_id'].long().to(self.device))
#             # attn_mask=x['clicked_title_pad'].to(self.device))

#         cdd_abs = x['candidate_abs'].long().to(self.device)
#         cdd_abs_embedding, cdd_abs_repr = self.encoder(
#             cdd_abs,
#             user_index=x['user_index'].long().to(self.device),
#             news_id=x['cdd_id'].long().to(self.device))
#             # attn_mask=x['candidate_abs_pad'].to(self.device))

#         his_abs = x['clicked_abs'].long().to(self.device)
#         his_abs_embedding, his_abs_repr = self.encoder(
#             his_abs,
#             user_index=x['user_index'].long().to(self.device),
#             news_id=x['his_id'].long().to(self.device))
#             # attn_mask=x['clicked_abs_pad'].to(self.device))

#         output_title = self.hisSelector(
#             cdd_title_repr, his_title_repr, his_title_embedding)

#         output_abs = self.hisSelector(
#             cdd_abs_repr, his_abs_repr, his_abs_embedding)

#         fusion_tensors_title = self.ranker(cdd_title_embedding, output_title[0])
#         fusion_tensors_abs = self.ranker(cdd_abs_embedding, output_abs[0])

#         # [bs, cs, 2, hd]
#         fusion_tensors = torch.tanh(torch.stack([self.title2view(fusion_tensors_title), self.abs2view(fusion_tensors_abs)], dim=-2))
#         fusion_tensor = Attention.ScaledDpAttention(self.viewQuery, fusion_tensors, fusion_tensors).squeeze(dim=-2)
#         return self._click_predictor(fusion_tensor)

#     def forward(self, x):
#         score = self._forward(x)
#         if self.training:
#             score = nn.functional.log_softmax(score, dim=1)
#         else:
#             score = torch.sigmoid(score)
#         return score