def _train(self, model, dataloader, optimizer, loss_func, epochs, scheduler=None):
        """ train model and print loss meanwhile
        Args:
            dataloader(torch.utils.data.DataLoader): provide data
            optimizer(list of torch.nn.optim): optimizer for training
            loss_func(torch.nn.Loss): loss function for training
            epochs(int): training epochs
            schedulers
            writer(torch.utils.tensorboard.SummaryWriter): tensorboard writer
            interval(int): within each epoch, the interval of training steps to display loss
            save_step(int): how many steps to save the model
        Returns:
            model: trained model
        """
        steps = 0
        interval = self.interval
        save_step = self.step
        distributed = self.world_size > 1

        if self.rank in [0, -1]:
            logger.info("training...{}".format(self.name))
            logger.info("total training step: {}".format(self.epochs * len(dataloader)))

        # if self.tb:
        #     writer = SummaryWriter("data/tb/{}/{}/{}/".format(
        #         self.name, self.scale, datetime.now().strftime("%Y%m%d-%H")))

        for epoch in range(epochs):
            epoch_loss = 0
            if distributed:
                dataloader.sampler.set_epoch(epoch)
            tqdm_ = tqdm(dataloader, smoothing=self.smoothing, ncols=120, leave=True)

            for step, x in enumerate(tqdm_):

                optimizer.zero_grad(set_to_none=True)

                pred = model(x)[0]
                label = x["label"].to(model.device)

                loss = loss_func(pred, label)

                epoch_loss += float(loss)

                loss.backward()

                optimizer.step()

                if scheduler:
                    scheduler.step()

                if step % interval == 0:
                    tqdm_.set_description(
                        "epoch: {:d}  step: {:d} total_step: {:d}  loss: {:.4f}".format(epoch + 1, step, steps, epoch_loss / step))
                    # if writer:
                    #     for name, param in model.named_parameters():
                    #         writer.add_histogram(name, param, step)

                    #     writer.add_scalar("data_loss",
                    #                     total_loss/total_steps)

                if save_step:
                    if steps % save_step == 0 and steps > 0:
                        self.save(model, steps, optimizer)

                steps += 1


    def train(self, model, loaders, tb=False):
        """ wrap training process

        Args:
            model(torch.nn.Module): the model to be trained
            loaders(list): list of torch.utils.data.DataLoader
            en: shell parameter
        """
        model.train()

        if self.rank in [0,-1]:
            # in case the folder does not exists, create one
            os.makedirs("data/model_params/{}".format(self.name), exist_ok=True)

        loss_func = self._get_loss(model)
        optimizer, scheduler = self._get_optim(model, len(loaders[0]))

        if self.checkpoint:
            self.load(model, self.checkpoint, optimizer)

        self._train(model, loaders[0], optimizer, loss_func, self.epochs, scheduler=scheduler)


max_input = {
            "cdd_id": torch.empty(1, self.impr_size).random_(0,10),
            "his_id": torch.empty(1, self.his_size).random_(0,10),
            "user_id": torch.tensor([1]),
            "cdd_encoded_index": torch.empty(1, self.impr_size, self.signal_length, dtype=torch.long).random_(0,10),
            "his_encoded_index": torch.empty(1, self.his_size, self.signal_length, dtype=torch.long).random_(0,10),
            "cdd_attn_mask": torch.ones(1, self.impr_size, self.signal_length),
            "his_attn_mask": torch.ones(1, self.his_size, self.signal_length),
            "cdd_subword_index": torch.ones(1, self.impr_size, self.signal_length, 2, dtype=torch.int64),
            "his_subword_index": torch.ones(1, self.his_size, self.signal_length, 2, dtype=torch.int64),
            "cdd_mask": torch.ones((1, self.impr_size, 1)),
            "his_mask": torch.ones((1, self.his_size, 1)),
        }
        if self.reducer == "matching":
            max_input["his_refined_mask"] = torch.ones(1, self.his_size, self.signal_length)
        elif self.reducer == "bow":
            max_input["cdd_encoded_index"] = torch.rand(1, self.impr_size, self.signal_length, 2, dtype=torch.long).random_(0,10)
            max_input["his_encoded_index"] = torch.rand(1, self.his_size, self.signal_length, 2, dtype=torch.long).random_(0,10)
            max_input["his_refined_mask"] = max_input["his_attn_mask"]
        elif self.reducer in ["bm25", "entity", "first"]:
            max_input["his_encoded_index"] = max_input["his_encoded_index"][:, :, :self.k+1]
            max_input["his_attn_mask"] = max_input["his_attn_mask"][:, :, :self.k+1]
            max_input["his_subword_index"] = max_input["his_subword_index"][:, :, :self.k+1]

        model(max_input)