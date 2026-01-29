import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer
import timm 

class ModelFactory:
    @staticmethod
    def get_model(model_name, num_classes=10):
        model = None
        
        # ==================================================
        # 1-10. Classic CNN models
        # ==================================================
        if model_name.startswith('resnet'):
            # Base model loading
            base_name = 'resnet18' if 'resnet18' in model_name else 'resnet50'
            if base_name == 'resnet18':
                model = models.resnet18(weights=None)
            else:
                model = models.resnet50(weights=None)
                
            # CIFAR small image adaptation
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()

            # Get feature dimension
            feat_dim = model.fc.in_features

            # Add normalization layer variants
            if model_name.endswith('_ln'):
                # ResNet + LayerNorm
                print(f">>> Constructing {model_name}: Adding LayerNorm before FC")
                model.fc = nn.Sequential(
                    nn.LayerNorm(feat_dim),
                    nn.Linear(feat_dim, num_classes, bias=False)
                )
            elif model_name.endswith('_bn'):
                # ResNet + BatchNorm
                print(f">>> Constructing {model_name}: Adding BatchNorm1d before FC")
                model.fc = nn.Sequential(
                    nn.BatchNorm1d(feat_dim),
                    nn.Linear(feat_dim, num_classes, bias=False)
                )
            else:
                # Standard ResNet (no normalization)
                model.fc = nn.Linear(feat_dim, num_classes, bias=False)
                
            model.features_extractor = model.avgpool

        elif model_name.startswith('vgg'):
            if model_name == 'vgg11': model = models.vgg11_bn(weights=None)
            elif model_name == 'vgg16': model = models.vgg16_bn(weights=None)
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.classifier = nn.Linear(512, num_classes, bias=False)
            model.fc = model.classifier

        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None)
            model.classifier = nn.Sequential(nn.Linear(model.last_channel, num_classes, bias=False))
            model.fc = model.classifier[0]

        elif model_name == 'shufflenet_v2':
            model = models.shufflenet_v2_x1_0(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)

        elif model_name == 'regnet_y_400mf':
             model = models.regnet_y_400mf(weights=None)
             model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)

        elif model_name == 'densenet121':
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes, bias=False)
            model.fc = model.classifier

        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Sequential(nn.Linear(1280, num_classes, bias=False))
            model.fc = model.classifier[0]
            
        elif model_name == 'googlenet':
            model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)

        elif model_name == 'convnext_tiny':
            model = models.convnext_tiny(weights=None)
            # ConvNeXt LayerNorm is in classifier[0]
            # We rewrite the entire classifier, removing the original LN
            dim = model.classifier[2].in_features
            model.classifier = nn.Sequential(nn.Flatten(1), nn.BatchNorm1d(dim, affine=True), nn.Linear(dim, num_classes, bias=False))
            model.fc = model.classifier[2]

        # ==================================================
        # ViT and variants
        # ==================================================
        elif model_name == 'vit_tiny':
            # Experimental group: ViT with BN (BN-ViT)
            model = VisionTransformer(image_size=32, patch_size=4, num_layers=6, num_heads=8, hidden_dim=512, mlp_dim=2048, num_classes=num_classes, dropout=0.0, attention_dropout=0.0)

            # Remove encoder LayerNorm to enable NC
            # Otherwise features are forcibly normalized, preventing NC
            if hasattr(model.encoder, 'ln'):
                model.encoder.ln = nn.Identity()

            model.heads = nn.Sequential(
                nn.BatchNorm1d(512, affine=True),
                nn.Linear(512, num_classes, bias=False)
            )
            model.fc = model.heads[1]

        elif model_name == 'vit_tiny_orig':
            # Control group: Original ViT
            # Keep LayerNorm to demonstrate how it prevents NC
            model = VisionTransformer(image_size=32, patch_size=4, num_layers=6, num_heads=8, hidden_dim=512, mlp_dim=2048, num_classes=num_classes, dropout=0.0, attention_dropout=0.0)
            model.heads = nn.Sequential(
                nn.Linear(512, num_classes, bias=False)
            )
            model.fc = model.heads[0]

        # ==================================================
        # Modern Models (MobileNetV4, EVA-02, etc.)
        # ==================================================
        elif model_name == 'mobilenetv4_small':
            model = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=False, num_classes=num_classes)
            if hasattr(model.classifier, 'in_features'):
                dim = model.classifier.in_features
            else:
                dim = model.num_features
                if dim == 960: dim = 1280 
            
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(dim, affine=True),
                nn.Linear(dim, num_classes, bias=False)
            )
            model.fc = model.classifier[1]

        elif model_name == 'repvit_m1':
            model = timm.create_model('repvit_m1.dist_in1k', pretrained=False, num_classes=num_classes)
            try:
                if isinstance(model.head, nn.Sequential):
                    dim = model.head[0].in_features
                else:
                    dim = model.num_features
            except:
                dim = model.num_features

            model.head = nn.Sequential(
                nn.BatchNorm1d(dim, affine=True),
                nn.Linear(dim, num_classes, bias=False)
            )
            model.fc = model.head[1]

        elif model_name == 'dinov3_small':
            model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=num_classes, img_size=32)
            dim = model.num_features
            # DINOv2 must remove Norm to enable NC
            model.norm = nn.Identity()
            model.head = nn.Sequential(
                nn.BatchNorm1d(dim, affine=True),
                nn.Linear(dim, num_classes, bias=False)
            )
            model.fc = model.head[1]

        elif model_name == 'EVA-02':
            # Explicitly add global_pool='token' to force CLS token usage
            model = timm.create_model(
                'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
                pretrained=False,
                num_classes=num_classes,
                img_size=32,
                global_pool='token'
            )
            model.norm = nn.Identity()

            dim = model.num_features
            model.head = nn.Sequential(
                nn.BatchNorm1d(dim, affine=True),
                nn.Linear(dim, num_classes, bias=False)
            )
            model.fc = model.head[1]

        elif model_name == 'EVA-02_orig':
            # Explicitly add global_pool='token'
            model = timm.create_model(
                'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
                pretrained=False,
                num_classes=num_classes,
                img_size=32,
                global_pool='token'
            )
            dim = model.num_features
            model.head = nn.Sequential(
                nn.BatchNorm1d(dim, affine=True),
                nn.Linear(dim, num_classes, bias=False)
            )
            model.fc = model.head[1]
        else:
            raise ValueError(f"Model {model_name} not supported.")
            
        return model

# =========================================================
# Feature extraction function
# =========================================================
def get_feature(model, x, model_name):
    # --- 1. ViT models ---
    if model_name == 'vit_tiny':
        x = model._process_input(x)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)  # encoder.ln replaced with Identity
        x = x[:, 0]
        if hasattr(model.heads, '0') and isinstance(model.heads[0], nn.BatchNorm1d):
            x = model.heads[0](x)
        return x

    if model_name == 'vit_tiny_orig':
        x = model._process_input(x)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)  # LayerNorm still present
        x = x[:, 0]
        return x

    # --- 2. Modern Models ---
    if model_name in ['dinov3_small', 'EVA-02', 'EVA-02_orig']:
        x = model.forward_features(x)
        # For EVA and DINO, we force global_pool='token' in ModelFactory
        # So forward_features returns [B, N, C], extract x[:, 0] for CLS token
        if model_name in ['EVA-02', 'EVA-02_orig']:
             x = x[:, 0]
        elif model.global_pool:
             x = x[:, 0] if model.global_pool == 'token' else x.mean(1)

        # Extract BN layer output from head (NC analysis needs normalized features)
        if hasattr(model, 'head') and isinstance(model.head[0], nn.BatchNorm1d):
            x = model.head[0](x)
        return x

    if model_name == 'mobilenetv4_small':
        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=True) 
        if isinstance(model.classifier[0], nn.BatchNorm1d): x = model.classifier[0](x)
        return x

    if model_name == 'repvit_m1':
        x = model.forward_features(x)
        x = model.forward_head(x, pre_logits=True)
        if isinstance(model.head[0], nn.BatchNorm1d): x = model.head[0](x)
        return x

    if model_name == 'convnext_tiny':
        x = model.features(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        if isinstance(model.classifier[1], nn.BatchNorm1d): x = model.classifier[1](x)
        return x

    # --- 3. Classic CNNs ---

    # ResNet must be handled separately
    if model_name.startswith('resnet'):
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    # RegNet handled separately to avoid AttributeError
    if model_name.startswith('regnet'):
        x = model.stem(x)
        x = model.trunk_output(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)
        
    if model_name.startswith('vgg'):
        x = model.features(x); x = model.avgpool(x)
        return torch.flatten(x, 1)

    elif model_name == 'mobilenet_v2':
        x = model.features(x); x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)
        
    elif model_name == 'shufflenet_v2':
        x = model.conv1(x); x = model.maxpool(x); x = model.stage2(x); x = model.stage3(x); x = model.stage4(x); x = model.conv5(x); x = x.mean([2, 3]) 
        return x
    
    elif model_name == 'densenet121':
        features = model.features(x); out = F.relu(features, inplace=True); out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    elif model_name == 'efficientnet_b0':
        x = model.features(x); x = model.avgpool(x)
        return torch.flatten(x, 1)

    elif model_name == 'googlenet':
        x = model.conv1(x); x = model.maxpool1(x); x = model.conv2(x); x = model.conv3(x); x = model.maxpool2(x)
        x = model.inception3a(x); x = model.inception3b(x); x = model.maxpool3(x); x = model.inception4a(x); x = model.inception4b(x); x = model.inception4c(x)
        x = model.inception4d(x); x = model.inception4e(x); x = model.maxpool4(x); x = model.inception5a(x); x = model.inception5b(x); x = model.avgpool(x)
        x = torch.flatten(x, 1)
        if model.dropout is not None: x = model.dropout(x)
        return x

    else:
        # Fallback: return logits directly if no match above
        return model(x)