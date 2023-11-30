import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHE_LoRA(nn.Module):
    def __init__(self, model):
        super(MHE_LoRA, self).__init__()
        # self.model = copy.deepcopy(model)
        self.model = self.copy_without_grad(model)

        self.extracted_params = {}
        keys_to_delete = []
        # for name, param in self.model.named_parameters():
        #     self.extracted_params[name] = param

        for name, tensor in model.state_dict().items():
            self.extracted_params[name] = tensor.detach().clone()

        for name in self.extracted_params:
            if 'attn' in name and 'processor' not in name:
                if 'weight' in name:
                    if 'to_q' in name:
                        lora_down = name.replace('to_q', 'processor.to_q_lora.down')
                        lora_up = name.replace('to_q', 'processor.to_q_lora.up')
                    elif 'to_k' in name:
                        lora_down = name.replace('to_k', 'processor.to_k_lora.down')
                        lora_up = name.replace('to_k', 'processor.to_k_lora.up')
                    elif 'to_v' in name:
                        lora_down = name.replace('to_v', 'processor.to_v_lora.down')
                        lora_up = name.replace('to_v', 'processor.to_v_lora.up')
                    elif 'to_out' in name:
                        lora_down = name.replace('to_out.0', 'processor.to_out_lora.down')
                        lora_up = name.replace('to_out.0', 'processor.to_out_lora.up')
                    else:
                        pass
                    with torch.no_grad():
                        self.extracted_params[name] += self.extracted_params[lora_up].cuda(
                        ) @ self.extracted_params[lora_down].cuda()
                    keys_to_delete.append(lora_up)
                    keys_to_delete.append(lora_down)

        for key in keys_to_delete:
            del self.extracted_params[key]

    def copy_without_grad(self, model):
        copied_model = copy.deepcopy(model)
        for param in copied_model.parameters():
            param.requires_grad = False
            param.detach_()
        return copied_model

    @staticmethod
    def mhe_loss(filt):
        if len(filt.shape) == 2:
            n_filt, _ = filt.shape
            filt = torch.transpose(filt, 0, 1)
            filt_neg = filt * (-1)
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

            filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
            norm_mat = torch.matmul(filt_norm.t(), filt_norm)
            inner_pro = torch.matmul(filt.t(), filt)
            inner_pro /= norm_mat

            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.tensor([1.0] * n_filt)).cuda())
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
            MHE_loss = 1 * torch.sum(final) / cnt

        else:
            n_filt, _, _, _ = filt.shape
            filt = filt.reshape(n_filt, -1)
            filt = torch.transpose(filt, 0, 1)
            filt_neg = filt * -1
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

            filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
            norm_mat = torch.matmul(filt_norm.t(), filt_norm)
            inner_pro = torch.matmul(filt.t(), filt)
            inner_pro /= norm_mat

            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.tensor([1.0] * n_filt)).cuda())
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
            MHE_loss = 1 * torch.sum(final) / cnt

        return MHE_loss

    def calculate_mhe(self):
        mhe_loss = []
        with torch.no_grad():
            for name in self.extracted_params:
                weight = self.extracted_params[name]
                # linear layer or conv layer
                if len(weight.shape) == 2 or len(weight.shape) == 4:
                    loss = self.mhe_loss(weight)
                    mhe_loss.append(loss.cpu().detach().item())
            mhe_loss = np.array(mhe_loss)
        return mhe_loss.sum()


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)


def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device,
                    dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


class MHE_OFT(nn.Module):
    def __init__(self, model, eps=2e-5, rank=4):
        super(MHE_OFT, self).__init__()
        # self.model = copy.deepcopy(model)
        # self.model = self.copy_without_grad(model)

        self.r = rank

        self.extracted_params = {}
        keys_to_delete = []
        # for name, param in self.model.named_parameters():
        #     self.extracted_params[name] = param

        for name, tensor in model.state_dict().items():
            self.extracted_params[name] = tensor.detach().clone()

        for name in self.extracted_params:
            if 'attn' in name and 'processor' not in name:
                if 'weight' in name:
                    if 'to_q' in name:
                        oft_R = name.replace('to_q.weight', 'processor.to_q_oft.R')
                    elif 'to_k' in name:
                        oft_R = name.replace('to_k.weight', 'processor.to_k_oft.R')
                    elif 'to_v' in name:
                        oft_R = name.replace('to_v.weight', 'processor.to_v_oft.R')
                    elif 'to_out' in name:
                        oft_R = name.replace('to_out.0.weight', 'processor.to_out_oft.R')
                    else:
                        pass

                    R = self.extracted_params[oft_R].cuda()

                    with torch.no_grad():
                        if len(R.shape) == 2:
                            self.eps = eps * R.shape[0] * R.shape[0]
                            R.copy_(project(R, eps=self.eps))
                            orth_rotate = self.cayley(R)
                        else:
                            self.eps = eps * R.shape[1] * R.shape[0]
                            R.copy_(project_batch(R, eps=self.eps))
                            orth_rotate = self.cayley_batch(R)

                        self.extracted_params[
                            name] = self.extracted_params[name] @ self.block_diagonal(orth_rotate)
                    keys_to_delete.append(oft_R)

        for key in keys_to_delete:
            del self.extracted_params[key]

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def block_diagonal(self, R):
        if len(R.shape) == 2:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(R.shape[0])]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def copy_without_grad(self, model):
        copied_model = copy.deepcopy(model)
        for param in copied_model.parameters():
            param.requires_grad = False
            param.detach_()
        return copied_model

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q

    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        # I = torch.eye(r, device=data.device).unsqueeze(0).repeat(b, 1, 1)
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I + skew, torch.inverse(I - skew))

        return Q

    @staticmethod
    def mhe_loss(filt):
        if len(filt.shape) == 2:
            n_filt, _ = filt.shape
            filt = torch.transpose(filt, 0, 1)
            filt_neg = filt * (-1)
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

            filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
            norm_mat = torch.matmul(filt_norm.t(), filt_norm)
            inner_pro = torch.matmul(filt.t(), filt)
            inner_pro /= norm_mat

            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.tensor([1.0] * n_filt)).cuda())
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
            MHE_loss = 1 * torch.sum(final) / cnt

        else:
            n_filt, _, _, _ = filt.shape
            filt = filt.reshape(n_filt, -1)
            filt = torch.transpose(filt, 0, 1)
            filt_neg = filt * -1
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

            filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
            norm_mat = torch.matmul(filt_norm.t(), filt_norm)
            inner_pro = torch.matmul(filt.t(), filt)
            inner_pro /= norm_mat

            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.tensor([1.0] * n_filt)).cuda())
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
            MHE_loss = 1 * torch.sum(final) / cnt

        return MHE_loss

    def calculate_mhe(self):
        mhe_loss = []
        with torch.no_grad():
            for name in self.extracted_params:
                weight = self.extracted_params[name]
                # linear layer or conv layer
                if len(weight.shape) == 2 or len(weight.shape) == 4:
                    loss = self.mhe_loss(weight)
                    mhe_loss.append(loss.cpu().detach().item())
            mhe_loss = np.array(mhe_loss)
        return mhe_loss.sum()

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))


class MHE_db:
    def __init__(self, model):
        # self.model = copy.deepcopy(model)
        # self.model.load_state_dict(model.state_dict())
        # self.model = self.copy_without_grad(model)

        #self.extracted_params = {}
        #for name, param in model.named_parameters():
        #    self.extracted_params[name] = param

        self.extracted_params = {}
        for name, tensor in model.state_dict().items():
            self.extracted_params[name] = tensor.detach().clone()

    @staticmethod
    def mhe_loss(filt):
        if len(filt.shape) == 2:
            n_filt, _ = filt.shape
            filt = torch.transpose(filt, 0, 1)
            filt_neg = filt * (-1)
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

            filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
            norm_mat = torch.matmul(filt_norm.t(), filt_norm)
            inner_pro = torch.matmul(filt.t(), filt)
            inner_pro /= norm_mat

            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.tensor([1.0] * n_filt)).cuda())
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
            MHE_loss = 1 * torch.sum(final) / cnt

        else:
            n_filt, _, _, _ = filt.shape
            filt = filt.reshape(n_filt, -1)
            filt = torch.transpose(filt, 0, 1)
            filt_neg = filt * -1
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

            filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
            norm_mat = torch.matmul(filt_norm.t(), filt_norm)
            inner_pro = torch.matmul(filt.t(), filt)
            inner_pro /= norm_mat

            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.tensor([1.0] * n_filt)).cuda())
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.tril(final)
            cnt = n_filt * (n_filt - 1) / 2.0
            MHE_loss = 1 * torch.sum(final) / cnt

        return MHE_loss

    def calculate_mhe(self):
        mhe_loss = []
        with torch.no_grad():
            for name in self.extracted_params:
                weight = self.extracted_params[name]
                # linear layer or conv layer
                if len(weight.shape) == 2 or len(weight.shape) == 4:
                    loss = self.mhe_loss(weight)
                    mhe_loss.append(loss.cpu().detach().item())
            mhe_loss = np.array(mhe_loss)
        return mhe_loss.sum()
