import torch
from einops import rearrange
from torch import nn

from .transformer import Transformer
from .weight_init import trunc_normal_

MIN_NUM_PATCHES = 16

class HybridEmbed(nn.Module):
    """ 
    Extract feature map from CNN, flatten, project to embedding dim.
    Based on rwightman's implementation:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, backbone, img_shape, embed_dim, feature_size=None):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        c, h, w = img_shape
        self.img_size = (h, w)
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                #
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, *img_shape))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size) if isinstance(feature_size, int) else feature_size
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.projection = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.projection(x)
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_shape, patch_size, embed_dim):
        super().__init__()

        c, h, w = img_shape
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert h % patch_size[0] == 0 and w % patch_size[1] == 0,\
            'image dimensions must be divisible by the patch size'

        self.patch_size = patch_size
        self.num_patches = (h // patch_size[0]) * (w // patch_size[1])
        patch_dim = c * patch_size[0] * patch_size[1]
        assert self.num_patches > MIN_NUM_PATCHES,\
            f'your number of patches ({self.num_patches}) is too small for ' \
            f'attention to be effective. try decreasing your patch size'

        self.projection = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        p1, p2 = self.patch_size

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2)
        x = self.projection(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, img_shape, patch_size, num_classes,
                 dim, depth, heads, mlp_dim, hybrid_backbone=None,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        if hybrid_backbone is not None:
            self.patch_embedding = HybridEmbed(hybrid_backbone, img_shape, dim)
        else:
            self.patch_embedding = PatchEmbed(img_shape, patch_size, dim)
        num_patches = self.patch_embedding.num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
