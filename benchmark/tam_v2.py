"""TAM v2 : propositions de correction pour benchmark_attention.py (NON TESTÉ : pas de torch dans mon environnement).

Contenu
  1. diagnose_old_tam : vérifier ce que fait réellement le TAM actuel (à lancer sur le checkpoint E4).
  2. TAMv2            : version corrigée (identité à l'init, texture Gabor à la bonne résolution, orientation conservée).
  3. make_optimizer   : LR plus élevé pour les modules neufs (à appliquer à TOUS les variants pour rester équitable).

Intégration dans benchmark_attention.py
  - Remplacer la ligne dans AttentionResNet50.__init__ :  "E4": TAMv2(2048)   (placement actuel, après layer4)
  - Variante recommandée (placement après layer3, carte 14x14, 1024 canaux) :
        self.attention = nn.Identity()  # pour E4 on n'utilise pas la position layer4
        self.tam3 = TAMv2(1024)
        # dans forward : x = self.layer3(x); x = self.tam3(x); x = self.layer4(x)
    (self.tam3 doit rester Identity pour E0..E3 ; le nom commence par "attention" ou adaptez make_optimizer.)
  - Dans train_one : optimizer = make_optimizer(model)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Diagnostic de l'ancien TAM
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diagnose_old_tam(model, loader, device, n_batches=5):
    """Affiche : taille de la carte, poids appris, amplitude de 'strength' et de la gate.
    Si gate.std() ~ 0 et gate.mean() ~ 0.5, le TAM est équivalent à x * 1.5 (donc à E0)."""
    model.eval()
    tam = model.attention
    print(f"proj.weight={tam.proj.weight.item():.5f}  proj.bias={tam.proj.bias.item():.5f}")
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        f = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        f = model.layer4(model.layer3(model.layer2(model.layer1(f))))
        tex = f.mean(1, keepdim=True)
        resp = F.conv2d(tex, tam.gabor_bank, padding=tam.gabor_bank.shape[-1] // 2)
        s = resp.abs().mean(1, keepdim=True)
        g = torch.sigmoid(tam.proj(s))
        print(f"batch {i}: feature map={tuple(f.shape[-2:])}  kernel={tuple(tam.gabor_bank.shape[-2:])}  "
              f"texture mean={tex.mean():.3f}  strength mean={s.mean():.5f}  "
              f"gate mean={g.mean():.4f} std={g.std():.5f}")


# --------------------------------------------------------------------------- #
# 2. TAM v2
# --------------------------------------------------------------------------- #
def make_gabor_bank_v2(k=5, sigma=1.5, wavelength=3.0, gamma=0.5, n_orient=8):
    """Banc de Gabor (n_orient, 1, k, k), moyenne nulle, normalisé L1.
    Valeurs par défaut adaptées à une carte 14x14 (layer3). Pour 28x28 : k=7, sigma=2, wavelength=4."""
    if k % 2 == 0:
        raise ValueError("k doit être impair")
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernels = []
    for theta in np.linspace(0, np.pi, n_orient, endpoint=False):
        xr = xx * np.cos(theta) + yy * np.sin(theta)
        yr = -xx * np.sin(theta) + yy * np.cos(theta)
        g = torch.exp(-(xr ** 2 + gamma ** 2 * yr ** 2) / (2 * sigma ** 2)) * torch.cos(2 * np.pi * xr / wavelength)
        g = g - g.mean()
        kernels.append(g / (g.abs().sum() + 1e-8))
    return torch.stack(kernels).unsqueeze(1)


class TAMv2(nn.Module):
    """Texture Attention Module v2.

    x (B,C,H,W) -> projection apprise C->m (au lieu de la moyenne sur les canaux)
                -> m*n_orient réponses de Gabor fixes (depthwise, orientation conservée)
                -> |.| -> petit MLP 1x1 -> carte spatiale tanh (B,1,H,W)
                -> sortie = x * (1 + gamma * carte), gamma initialisé à 0  => identité exacte à l'init.
    """

    def __init__(self, c, m=8, k=5, n_orient=8, **gabor_kw):
        super().__init__()
        bank = make_gabor_bank_v2(k=k, n_orient=n_orient, **gabor_kw)       # (O,1,k,k)
        self.register_buffer("gabor", bank.repeat(m, 1, 1, 1))              # (m*O,1,k,k), groups=m
        self.m, self.k = m, k
        self.reduce = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.BatchNorm2d(m))
        self.gate = nn.Sequential(
            nn.Conv2d(m * n_orient, m, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(inplace=True),
            nn.Conv2d(m, 1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        t = self.reduce(x)
        r = F.conv2d(t, self.gabor, padding=self.k // 2, groups=self.m).abs()
        a = torch.tanh(self.gate(r))
        return x * (1.0 + self.gamma * a)


# --------------------------------------------------------------------------- #
# 3. Optimiseur à LR différencié (à appliquer à tous les variants E0..E4)
# --------------------------------------------------------------------------- #
def make_optimizer(model, lr_backbone=1e-4, lr_new=1e-3, weight_decay=1e-4):
    new_prefixes = ("attention", "tam3", "fc")
    new, old = [], []
    for n, p in model.named_parameters():
        (new if n.startswith(new_prefixes) else old).append(p)
    return torch.optim.AdamW(
        [{"params": old, "lr": lr_backbone}, {"params": new, "lr": lr_new}],
        weight_decay=weight_decay,
    )
