import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_h=256, max_w=256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        assert d_model % 2 == 0
        d_half = d_model // 2
        pe_h = torch.zeros(max_h, d_half)
        pe_w = torch.zeros(max_w, d_half)
        pos_h = torch.arange(0, max_h).unsqueeze(1)
        pos_w = torch.arange(0, max_w).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_half, 2) * -(math.log(10000.0) / d_half))
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        self.register_buffer("pe_h", pe_h)
        self.register_buffer("pe_w", pe_w)
        self.scale = nn.Parameter(torch.ones(1, d_model, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        pe_h = self.pe_h[:H].unsqueeze(1).repeat(1, W, 1)
        pe_w = self.pe_w[:W].unsqueeze(0).repeat(H, 1, 1)
        pe = torch.cat([pe_h, pe_w], dim=-1)
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        return self.dropout(self.scale * pe)


class Interactor(nn.Module):
    def __init__(self, dim_x, dim_y, num_heads=8, depth=2):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleDict({
                'proj_y_to_x': nn.Linear(dim_y, dim_x),
                'proj_x_to_y': nn.Linear(dim_x, dim_y),
                'attn_x': nn.MultiheadAttention(dim_x, num_heads, batch_first=True),
                'attn_y': nn.MultiheadAttention(dim_y, num_heads, batch_first=True),
                'norm_x': nn.LayerNorm(dim_x),
                'norm_y': nn.LayerNorm(dim_y)
            })
            self.layers.append(layer)

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        for layer in self.layers:
            y_proj = layer['proj_y_to_x'](y)
            x_upd, _ = layer['attn_x'](x, y_proj, y_proj)
            x = layer['norm_x'](x + x_upd)
            x_proj = layer['proj_x_to_y'](x)
            y_upd, _ = layer['attn_y'](y, x_proj, x_proj)
            y = layer['norm_y'](y + y_upd)
        return x, y


class DSEM(nn.Module):
    def __init__(self, dim, boundary_type='periodic', use_viscosity=True, cfl_factor=0.5):
        super().__init__()
        self.boundary_type = boundary_type
        self.use_viscosity = use_viscosity

        self.pos_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid()
        )
        self.flow_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim * 2)
        )
        self.text_align = nn.Linear(768, dim)
        self.text_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.cfl_logit = nn.Parameter(torch.logit(torch.tensor(cfl_factor)))
        self.source_strength_logit = nn.Parameter(torch.tensor(0.1))
        self.viscosity_logit = nn.Parameter(torch.tensor(0.01))

        self.diff_scale = nn.Parameter(torch.tensor(1.5))

        self.register_buffer('dx', torch.tensor(1.0))
        self.register_buffer('dy', torch.tensor(1.0))

        kernel_size = 3
        sigma = 0.5
        kernel = self._gaussian_kernel(kernel_size, sigma, device='cpu')
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        self.register_buffer("gaussian_kernel", kernel)
        self.kernel_size = kernel_size

    def _get_cfl(self):
        return torch.sigmoid(self.cfl_logit) * 0.99 + 0.01

    def _pad(self, x, l, r, t, b):
        if self.boundary_type == 'periodic':
            return F.pad(x, (l, r, t, b), mode='circular')
        else:
            mode = 'reflect' if self.boundary_type == 'neumann' else 'replicate'
            return F.pad(x, (l, r, t, b), mode=mode)

    def _gaussian_kernel(self, size, sigma, device):
        coords = torch.arange(size, device=device) - size // 2
        grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), dim=-1)
        kernel = torch.exp(-torch.sum(grid ** 2, dim=-1) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x, text, pos, H, W, gamma=None):
        B, N, C = x.shape
        x_map = x.transpose(1, 2).reshape(B, C, H, W)

        pos_feat = self.pos_proj(pos)
        flow = torch.tanh(self.flow_proj(pos_feat))
        flow = flow.transpose(1, 2).reshape(B, 2 * C, H, W)
        vx, vy = flow.chunk(2, dim=1)

        text_feat = self.text_align(text)
        attn = torch.softmax((x @ text_feat.transpose(1, 2)) / (C ** 0.5), dim=-1)
        source = self.text_proj(attn @ text_feat)
        source = source.transpose(1, 2).reshape(B, C, H, W)

        dx_min = min(self.dx, self.dy)

        v_abs = torch.maximum(torch.abs(vx), torch.abs(vy))
        v_max = v_abs.amax(dim=(2,3), keepdim=True).amax(dim=1, keepdim=True) + 1e-6

        cfl = self._get_cfl()
        dt_adv = cfl * dx_min / v_max

        visc_val = torch.sigmoid(self.viscosity_logit) * 0.1
        dt_diff = 0.5 * dx_min ** 2 / (visc_val + 1e-6)

        dt = torch.min(dt_adv, dt_diff)
        dt = torch.clamp(dt, max=0.05)

        dx_diff = x_map[:, :, :, 1:] - x_map[:, :, :, :-1]
        gx_l = self._pad(dx_diff, 1, 0, 0, 0)
        gx_r = self._pad(dx_diff, 0, 1, 0, 0)

        dy_diff = x_map[:, :, 1:, :] - x_map[:, :, :-1, :]
        gy_t = self._pad(dy_diff, 0, 0, 1, 0)
        gy_b = self._pad(dy_diff, 0, 0, 0, 1)
        eps = 1e-6
        r_x = gx_l / (gx_r + eps)
        r_y = gy_t / (gy_b + eps)
        r_x = torch.tanh(r_x)
        r_y = torch.tanh(r_y)
        phi_x = 0.5 * (1 + r_x)
        phi_y = 0.5 * (1 + r_y)
        gx_lim = phi_x * gx_l + (1 - phi_x) * gx_r
        gy_lim = phi_y * gy_t + (1 - phi_y) * gy_b
        smooth_x = 0.25 * (gx_l + gx_r)
        smooth_y = 0.25 * (gy_t + gy_b)
        grad_mag = torch.sqrt(gx_l ** 2 + gy_t ** 2 + 1e-6)
        edge_weight = torch.sigmoid(5 * (grad_mag - grad_mag.mean()))

        gx = edge_weight * gx_lim + (1 - edge_weight) * smooth_x
        gy = edge_weight * gy_lim + (1 - edge_weight) * smooth_y

        adv = vx * gx + vy * gy

        source_strength = torch.sigmoid(self.source_strength_logit)
        relax = source_strength * (source - x_map)

        if gamma is not None:
            gamma_mean = gamma.mean(dim=(1, 2), keepdim=True).view(B, 1, 1, 1)
            relax = relax + 0.1 * gamma_mean * (source - x_map)

        if self.use_viscosity and H >= 2 and W >= 2:
            grad_x = x_map[:, :, :, 1:] - x_map[:, :, :, :-1]
            grad_y = x_map[:, :, 1:, :] - x_map[:, :, :-1, :]

            grad_x = self._pad(grad_x, 1, 0, 0, 0)
            grad_y = self._pad(grad_y, 0, 0, 1, 0)

            Sxx = grad_x * grad_x
            Syy = grad_y * grad_y
            Sxy = grad_x * grad_y

            kernel = self.gaussian_kernel.to(x_map.device)
            depthwise_kernel = kernel.repeat(C, 1, 1, 1)

            Sxx = F.conv2d(Sxx, depthwise_kernel, padding=self.kernel_size // 2, groups=C)
            Syy = F.conv2d(Syy, depthwise_kernel, padding=self.kernel_size // 2, groups=C)
            Sxy = F.conv2d(Sxy, depthwise_kernel, padding=self.kernel_size // 2, groups=C)

            trace = Sxx + Syy
            disc = torch.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2 + 1e-6)

            lambda1 = (trace + disc) / 2
            lambda2 = (trace - disc) / 2

            lambda1_norm = lambda1 / (lambda1.mean(dim=(2, 3), keepdim=True) + 1e-6)
            lambda2_norm = lambda2 / (lambda2.mean(dim=(2, 3), keepdim=True) + 1e-6)

            c_perp = torch.exp(-(lambda1_norm) ** 2)
            c_para = torch.exp(-(lambda2_norm) ** 2)

            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            nx = grad_x / grad_mag
            ny = grad_y / grad_mag

            dot_n_grad = nx * grad_x + ny * grad_y

            flux_x = c_para * grad_x + (c_perp - c_para) * dot_n_grad * nx
            flux_y = c_para * grad_y + (c_perp - c_para) * dot_n_grad * ny

            flux_x_pad = self._pad(flux_x, 0, 1, 0, 0)
            flux_y_pad = self._pad(flux_y, 0, 0, 0, 1)

            aniso_diff = (flux_x_pad[:, :, :, 1:] - flux_x) + \
                         (flux_y_pad[:, :, 1:, :] - flux_y)

            aniso_diff = aniso_diff / (aniso_diff.std(dim=(2,3), keepdim=True) + 1e-6)

            diff_scale = torch.clamp(self.diff_scale, 0.5, 3.0)
            visc = diff_scale * visc_val * dx_min ** 2

            adv = adv + visc * aniso_diff

        x_map = x_map + dt * (adv + relax)

        x_new = x_map.reshape(B, C, -1).transpose(1, 2)

        x_norm = F.normalize(x_new, dim=-1, eps=1e-6)
        structure_mat = x_norm @ x_norm.transpose(-1, -2)

        return x_new, pos, structure_mat


class SCOT(nn.Module):
    def __init__(self, dim, max_iters=5, epsilon=0.5, token_ratio=0.5,
                 lambda_sem=1.0, lambda_spa=0.3, lambda_str=0.3, lambda_pde=0.2, sparse_k=5):
        super().__init__()
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.scale = dim ** -0.5
        self.token_ratio = token_ratio
        self.lambda_sem = lambda_sem
        self.lambda_spa = lambda_spa
        self.lambda_str = lambda_str
        self.lambda_pde = lambda_pde
        self.sparse_k = sparse_k

    def sinkhorn(self, cost):
        B, N, T = cost.shape
        cost = torch.clamp(cost, -10, 10)
        K = torch.exp(-cost / self.epsilon).clamp(min=1e-8, max=1e8)
        mu = torch.full((B, N), 1.0 / N, device=cost.device)
        nu = torch.full((B, T), 1.0 / T, device=cost.device)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for _ in range(self.max_iters):
            u = mu / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1e8)
            v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp(1e-8, 1e8)
        transport = u.unsqueeze(-1) * K * v.unsqueeze(1)
        transport = transport / (transport.sum(dim=-1, keepdim=True) + 1e-8)
        return transport

    def compress_tokens(self, tokens):
        if self.token_ratio >= 1.0:
            return tokens
        B, N, C = tokens.shape
        target_len = max(int(N * self.token_ratio), 16)
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), target_len).transpose(1, 2)

    def compress_structure(self, structure_mat, target_len):
        B, N, _ = structure_mat.shape
        structure_reshaped = structure_mat.view(B, 1, N, N)
        compressed = F.adaptive_avg_pool2d(structure_reshaped, (target_len, target_len))
        return compressed.squeeze(1)

    def compute_cost(self, img_tokens, sem_tokens, structure_comp=None, img_coords=None):
        B, N, _ = img_tokens.shape
        T = sem_tokens.shape[1]

        sim = torch.matmul(img_tokens, sem_tokens.transpose(-1, -2)) * self.scale
        semantic_cost = -sim

        if img_coords is None:
            H = int(math.sqrt(N))
            W = H
            grid_h = torch.linspace(-1, 1, H, device=img_tokens.device)
            grid_w = torch.linspace(-1, 1, W, device=img_tokens.device)
            grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
            coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)  # [N,2]
            coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B,N,2]
        else:
            coords = img_coords
        txt_coords = torch.zeros(B, T, 2, device=img_tokens.device)
        spatial = torch.norm(coords.unsqueeze(2) - txt_coords.unsqueeze(1), dim=-1)  # [B,N,T]
        spatial = spatial / math.sqrt(2)

        structure_old = torch.zeros(B, N, T, device=img_tokens.device)

        cost = (self.lambda_sem * semantic_cost +
                self.lambda_spa * spatial +
                self.lambda_str * structure_old)

        if structure_comp is not None:
            sem_sim = torch.matmul(sem_tokens, sem_tokens.transpose(-1, -2)) * self.scale
            sem_sim_comp = F.adaptive_avg_pool2d(sem_sim.unsqueeze(1),
                                                 (structure_comp.size(1), structure_comp.size(2))).squeeze(1)
            pde_cost = F.mse_loss(sem_sim_comp, structure_comp, reduction='none').mean(dim=(1, 2))
            pde_cost = pde_cost.unsqueeze(-1).unsqueeze(-1).expand(B, N, T)
            cost = cost + self.lambda_pde * pde_cost

        return cost.clamp(-10, 10)

    def forward(self, img_tokens, sem_tokens, structure_mat=None, img_coords=None):
        B, N, C = img_tokens.shape
        img_comp = self.compress_tokens(img_tokens)
        N_comp = img_comp.size(1)

        structure_comp = None
        if structure_mat is not None:
            structure_comp = self.compress_structure(structure_mat, N_comp)

        if img_coords is not None:
            img_coords_comp = F.adaptive_avg_pool1d(img_coords.transpose(1, 2), N_comp).transpose(1, 2)
        else:
            img_coords_comp = None

        cost = self.compute_cost(img_comp, sem_tokens, structure_comp, img_coords_comp)
        transport = self.sinkhorn(cost)

        k = min(self.sparse_k, transport.size(-1))
        topk_vals, topk_idx = torch.topk(transport, k, dim=-1)
        mask = torch.zeros_like(transport).scatter_(-1, topk_idx, 1.0)
        transport = transport * mask
        transport = transport / (transport.sum(dim=-1, keepdim=True) + 1e-8)

        aligned_comp = torch.matmul(transport, sem_tokens)

        if N_comp != N:
            aligned = F.interpolate(aligned_comp.transpose(1, 2), size=N, mode='linear', align_corners=False).transpose(
                1, 2)
        else:
            aligned = aligned_comp

        return img_tokens + aligned, transport


class TextTokenReducer(nn.Module):
    def __init__(self, in_dim=768, num_tokens=196):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_tokens, in_dim))
        self.attn = nn.MultiheadAttention(in_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out)


class RegionTokenizer(nn.Module):
    def __init__(self, num_regions=16):
        super().__init__()
        self.grid = int(num_regions ** 0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        region = F.adaptive_avg_pool2d(x, self.grid)
        region = region.flatten(2).transpose(1, 2)
        return region


class MultiScaleOT(nn.Module):
    def __init__(self, in_c, embed_dim, num_regions=16):
        super().__init__()
        self.tokenizer = RegionTokenizer(num_regions)
        self.img_proj = nn.Linear(in_c, embed_dim)
        self.txt_proj = nn.Linear(768, embed_dim)
        self.ot = SCOT(embed_dim)
        self.recover = nn.Linear(embed_dim, in_c)

    def forward(self, x, text, structure_mat=None):
        B, C, H, W = x.shape
        tokens = self.tokenizer(x)
        tokens = self.img_proj(tokens)
        text = self.txt_proj(text)
        text = F.normalize(text, dim=-1)
        grid_h = torch.linspace(-1, 1, self.tokenizer.grid, device=x.device)
        grid_w = torch.linspace(-1, 1, self.tokenizer.grid, device=x.device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1)
        tokens, transport = self.ot(tokens, text, structure_mat, img_coords=coords)
        region_feats = tokens
        tokens = self.recover(tokens)
        grid = int(math.sqrt(tokens.size(1)))
        tokens = tokens.transpose(1, 2).view(B, C, grid, grid)
        x_rec = F.interpolate(tokens, size=(H, W), mode='bilinear', align_corners=False)
        return x_rec, region_feats, transport


class DualFiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_text = nn.Linear(cond_dim, feat_dim)
        self.beta_text = nn.Linear(cond_dim, feat_dim)

        self.gamma_struct = nn.Linear(cond_dim, feat_dim)
        self.beta_struct = nn.Linear(cond_dim, feat_dim)

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, text_cond, struct_cond):
        gamma_t = self.gamma_text(text_cond)
        beta_t = self.beta_text(text_cond)

        gamma_s = self.gamma_struct(struct_cond)
        beta_s = self.beta_struct(struct_cond)

        alpha = torch.tanh(self.alpha)

        gamma = gamma_t + alpha * gamma_s
        beta = beta_t + alpha * beta_s

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta

class PixLevelModule(nn.Module):
    def __init__(self, in_channels):
        super(PixLevelModule, self).__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1)
        x_avg = x_avg.unsqueeze(dim=1)
        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)
        x_out = x_max+x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        y = x_output * x
        return y


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', use_film=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.use_film = use_film

        if use_film:
            self.film = DualFiLM(cond_dim=768, feat_dim=in_channels // 2)

        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x, text_cond=None, struct_cond=None, structure_map=None):
        up = self.up(x)

        skip_att = self.pixModule(skip_x)

        if structure_map is not None:
            structure_map_resized = F.interpolate(structure_map,size=skip_att.shape[-2:],mode='bilinear',align_corners=False)
            structure_map_resized = torch.tanh(structure_map_resized)
            skip_att = skip_att * (1 + structure_map_resized)

        if self.use_film and text_cond is not None and struct_cond is not None:
            skip_att = self.film(skip_att, text_cond, struct_cond)

        x = torch.cat([skip_att, up], dim=1)
        return self.nConvs(x)


class QNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, in_channels=64, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)

        self.text_reducer = TextTokenReducer()
        self.pos_enc = PositionalEncoding2D(in_channels * 8)
        self.text_pos_interactor = Interactor(768, in_channels * 8, 8)
        self.img_text_interactor = Interactor(in_channels * 8, 768, 8)
        self.pde = DSEM(in_channels * 8)
        self.ot = MultiScaleOT(in_channels * 8, 256, num_regions=16)
        self.struct_pool_size = 16
        self.struct_encoder = nn.Sequential(
            nn.Linear(self.struct_pool_size * self.struct_pool_size, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
        self.text_pos_proj = nn.Linear(768, 768)
        self.text_img_proj = nn.Linear(768, 768)

        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2, use_film=True)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2, use_film=True)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2, use_film=True)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2, use_film=True)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, text=None):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        B, C, H, W = x5.shape

        text_red = self.text_reducer(text)
        pos = self.pos_enc(x5)
        pos = pos.flatten(2).transpose(1, 2)
        xb_tokens = x5.flatten(2).transpose(1, 2)
        pos = pos.expand(B, -1, -1)

        text_pos = self.text_pos_proj(text_red)
        t_p, p_t = self.text_pos_interactor(text_pos, pos)

        text_enhanced = text_red + t_p

        text_img = self.text_img_proj(text_enhanced)
        x_t, t_x = self.img_text_interactor(xb_tokens + p_t, text_img)

        x_pde, pos, structure_mat = self.pde(x_t, t_p, p_t, H, W, gamma=None)

        x_ot, region_feats, transport = self.ot(x_pde.reshape(x5.shape), t_p, structure_mat)
        B, N, _ = structure_mat.shape
        structure_pooled = F.adaptive_avg_pool2d(
            structure_mat.view(B, 1, N, N), (self.struct_pool_size, self.struct_pool_size)
        ).view(B, -1)
        structure_token = self.struct_encoder(structure_pooled)
        structure_map = structure_mat.mean(dim=1)
        structure_map = structure_map.view(B, 1, H, W)

        text_pool = t_p.mean(1)
        text_cond = text_pool
        struct_cond = structure_token

        x_ot_tokens = x_ot.flatten(2).transpose(1, 2)
        x_pde2, _, _ = self.pde(x_ot_tokens, t_p, p_t, H, W, gamma=None)
        x_final = x_pde2.transpose(1, 2).reshape(x5.shape)

        d4 = self.up4(x_final, x4, text_cond, struct_cond, structure_map)
        d3 = self.up3(d4, x3, text_cond, struct_cond, structure_map)
        d2 = self.up2(d3, x2, text_cond, struct_cond, structure_map)
        d1 = self.up1(d2, x1, text_cond, struct_cond, structure_map)

        logits = self.outc(d1)
        return logits

def test():
    model = QNet()
    img = torch.randn(4, 3, 224, 224)
    text = torch.randn(4, 32, 768)
    seg = model(img, text)
    print("seg shape:", seg.shape)

    flops, params = profile(model, inputs=(img,text))
    print(f"FLOPs: {flops / 1e9:.2f}G  Params: {params / 1e6:.2f}M")



if __name__ == "__main__":
    test()