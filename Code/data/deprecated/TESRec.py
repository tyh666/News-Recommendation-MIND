def _forward(self,x):
        # destroy encoding and embedding outside of the model

        if self.granularity != 'token':
            batch_size = x['cdd_subword_index'].size(0)
            cdd_size = x['cdd_subword_index'].size(1)

            if self.training:
                cdd_dest = self.cdd_dest[:batch_size, :cdd_size]
                his_dest = self.his_dest[:batch_size]

            # batch_size always equals 1 when evaluating
            else:
                cdd_dest = self.cdd_dest[[0], :cdd_size]
                his_dest = self.his_dest[[0]]

            cdd_subword_index = x['cdd_subword_index'].to(self.device)
            his_subword_index = x['his_subword_index'].to(self.device)
            his_signal_length = his_subword_index.size(-2)
            cdd_subword_index = cdd_subword_index[:, :, :, 0] * self.signal_length + cdd_subword_index[:, :, :, 1]
            his_subword_index = his_subword_index[:, :, :, 0] * his_signal_length + his_subword_index[:, :, :, 1]

            if self.training:
                # * cdd_mask to filter out padded cdd news
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1) * x["cdd_mask"].to(self.device)
            else:
                cdd_subword_prefix = cdd_dest.scatter(dim=-1, index=cdd_subword_index, value=1)

            cdd_subword_prefix = cdd_subword_prefix.view(batch_size, cdd_size, self.signal_length, self.signal_length)
            his_subword_prefix = his_dest.scatter(dim=-1, index=his_subword_index, value=1) * x["his_mask"].to(self.device)
            his_subword_prefix = his_subword_prefix.view(batch_size, self.his_size, his_signal_length, his_signal_length)

            if self.granularity == 'avg':
                # average subword embeddings as the word embedding
                cdd_subword_prefix = F.normalize(cdd_subword_prefix, p=1, dim=-1)
                his_subword_prefix = F.normalize(his_subword_prefix, p=1, dim=-1)

            cdd_attn_mask = cdd_subword_prefix.matmul(x['cdd_attn_mask'].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_attn_mask = his_subword_prefix.matmul(x["his_attn_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = his_subword_prefix.matmul(x["his_refined_mask"].to(self.device).float().unsqueeze(-1)).squeeze(-1)

        else:
            cdd_subword_prefix = None
            his_subword_prefix = None
            cdd_attn_mask = x['cdd_attn_mask'].to(self.device)
            his_attn_mask = x["his_attn_mask"].to(self.device)
            his_refined_mask = None
            if 'his_refined_mask' in x:
                his_refined_mask = x["his_refined_mask"].to(self.device)

        cdd_news = x["cdd_encoded_index"].to(self.device)
        _, cdd_news_repr, _ = self.bert(
            self.embedding(cdd_news, cdd_subword_prefix), cdd_attn_mask
        )

        his_news = x["his_encoded_index"].to(self.device)
        his_news_embedding = self.embedding(his_news, his_subword_prefix)
        if hasattr(self, 'encoderN'):
            his_news_encoded_embedding, his_news_repr = self.encoderN(
                his_news_embedding, his_attn_mask
            )
        else:
            his_news_encoded_embedding = None
            his_news_repr = None

        # no need to calculate this if ps_terms are fixed in advance
        if self.reducer.name == 'matching':
            user_repr = self.encoderU(his_news_repr, his_mask=x['his_mask'].to(self.device), user_index=x["user_id"].to(self.device))
        else:
            user_repr = None

        ps_terms, ps_term_mask, kid = self.reducer(his_news_encoded_embedding, his_news_embedding, user_repr, his_news_repr, his_attn_mask, his_refined_mask)

        _, user_repr, _ = self.bert(ps_terms, ps_term_mask, ps_term_input=True)

        # user_repr = self.newsUserProject(user_repr)

        if self.aggregator is not None:
            user_repr = self.aggregator(user_repr)

        if hasattr(self, 'userBias'):
            user_repr = user_repr + self.userBias

        return self.clickPredictor(cdd_news_repr, user_repr), kid

