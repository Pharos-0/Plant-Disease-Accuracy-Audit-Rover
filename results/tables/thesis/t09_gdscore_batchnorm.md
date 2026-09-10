**Table 9. GdScore MAE under train- vs eval-mode activations, 5 seeds per backbone. ViT has no BatchNorm and is the control.**

| backbone | eval | train | change | has BatchNorm |
|---|---|---|---|---|
| efficientnet_b0 | 0.4878 | 0.0736 | 0.4142 | yes |
| mobilenetv3_large | 0.4715 | 0.1021 | 0.3694 | yes |
| resnet50 | 0.4456 | 0.052 | 0.3936 | yes |
| vit_small_patch16 | 0.4122 | 0.412 | 0.0002 | no |
