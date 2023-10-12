# Trainer for MaskGIT
import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

from Trainer.trainer import Trainer
from Network.transformer import MaskTransformer

from Network.Taming.models.vqgan import VQModel


class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args
        self.patch_size = self.args.img_size // 16
        self.scaler = torch.cuda.amp.GradScaler()
        self.vit = self.get_network("vit")
        self.ae = self.get_network("autoencoder")
        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))
        if not self.args.debug:
            self.train_data, self.test_data = self.get_data()

        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            self.sae = SampleAndEval(device=self.args.device, num_images=50_000)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            model = MaskTransformer(
                img_size=self.args.img_size, hidden_dim=768, codebook_size=1024, depth=24, heads=16, mlp_dim=3072, dropout=0.1     # Small
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
            )

            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
                if self.args.is_master:
                    print("load ckpt from:", ckpt)
                checkpoint = torch.load(ckpt, map_location='cpu')
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model = model.to(self.args.device)
            # try:
            #     model = torch.compile(model)
            # except AttributeError:
            #     if self.args.is_master:
            #         print("cannot compile model --> update to pytorch 2.0")

            if self.args.is_multi_gpus:
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "autoencoder":
            config = OmegaConf.load(self.args.vqgan_folder + "model.yaml")
            model = VQModel(**config.model.params)
            checkpoint = torch.load(self.args.vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)
            # try:
            #     model = torch.compile(model)
            # except AttributeError:
            #     if self.args.is_master:
            #         print("cannot compile model --> update to pytorch 2.0")

            if self.args.is_multi_gpus:
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "square":
            val_to_mask = (r ** 2)
        elif mode == "cosine":
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        elif mode == "linear":
            val_to_mask = r
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        if value > 0:
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, 1024)

        return mask_code, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":
            val_to_mask = 1 - r
        elif mode == "square":
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return
        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up to 16*16=256
        return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=2500):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss = 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        for x, y in bar:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            x = 2 * (x - x.min()) / (x.max() - x.min()) - 1    # VQGAN take normalized img

            drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

            # VQGAN encoding to img tokens
            with torch.no_grad():
                emb, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask the encoded tokens
            masked_code, mask = self.get_mask_code(code, value=self.args.mask_value)

            with torch.cuda.amp.autocast():                             # half precision
                pred = self.vit(masked_code, y, drop_label=drop_label)  # The unmasked tokens prediction
                loss = self.criterion(pred.reshape(-1, 1024 + 1), code.view(-1)) / self.args.grad_cum

            update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()
            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            if update_grad and self.args.is_master:
                self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

            if self.args.iter % log_iter == 0 and self.args.is_master:
                # Generate sample for visualization
                gen_sample = self.sample(nb_sample=10)[0]
                gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                # Show reconstruction
                unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
                reco_sample = vutils.make_grid(reco_sample.data, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

                # Save Network
                self.save_network(model=self.vit, path=self.args.vit_folder+"current.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            self.args.iter += 1

        return cum_loss / n

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")
        start = time.time()
        for e in range(self.args.global_epoch, self.args.epoch):
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)
            train_loss = self.train_one_epoch()
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            if e % 10 == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', train_loss, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Loss {train_loss:.4f},"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1

    def eval(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        m = self.sae.compute_and_log_metrics(self)
        self.vit.train()
        return m

    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):

        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                _x = self.ae.decode_code(torch.clamp(code, 0, 1023))
                if mask is not None:
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0, 1023))
                l_visual.append(__x)

            if unmasked_code is not None:
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, 1023))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []
        l_mask = []
        with torch.no_grad():
            if labels is None:
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:
                code = init_code
                mask = (init_code == 1024).float().view(nb_sample, self.patch_size*self.patch_size)
            else:
                if self.args.mask_value < 0:
                    code = torch.randint(0, 1024, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            if isinstance(sched_mode, str):
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:
                scheduler = sched_mode

            for indice, t in enumerate(scheduler):
                if mask.sum() < t: t = int(mask.sum().item())
                if mask.sum() == 0: break

                with torch.cuda.amp.autocast():  # half precision
                    if w != 0:
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / len(scheduler))
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        logit = self.vit(code.clone(), labels, drop_label=~drop)

                prob = torch.softmax(logit * sm_temp, -1)
                distri = torch.distributions.Categorical(probs=prob)

                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / len(scheduler))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            _code = torch.clamp(code, 0, 1023)
            x = self.ae.decode_code(_code)
        self.vit.train()
        return x, l_codes, l_mask
