"""
Multimodal Network

processing 3D Image Data (VISION) and clinical data (tabular TEXT as vec of numbers).

ResNet + MLP (down transform) + RELU for Vision
MLP + RELU for Clinical Data
MLP for Fusion
Cls Head (dense) for Regression

Optional Embedding:
    Use with awareness!
"""
from typing import Sequence, Union

import torch
from monai.networks.blocks import MLPBlock
from monai.networks.nets import ResNet as ResNetMonai
from monai.networks.nets import ResNetBlock as ResNetBlock_MONAI
import torch.nn as nn
import numpy as np

class Regression(nn.Module):
    """
    Simple Linear Layer. No activation function.
    """

    def __init__(self, in_features, out_features=1) -> None:
        super().__init__()
        self.denseCLS = nn.Linear(in_features, out_features)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        cls_output = self.denseCLS(first_token_tensor)
        return cls_output

class TabularEmbedding(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()
        self.tabular_embedding = nn.Embedding(num_features, embedding_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, embedding_size))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward_embedding(self, x):
        #print("x_shape", x.shape)
        x = self.tabular_embedding(x)
        x = x + self.position_embedding
        # x_shape is [batch_size, x[1].shape=seq_len, embedding_size]
        #transform to [batch_size, embedding_size] with AdaptiveAvgPool1d
        #print("x.shape afer embedding", x.shape)
        x = x.transpose(1,2)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x):
        return self.forward_embedding(x)


class vision_head(nn.Module): # regression head
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self,x):
        return self.fc(x)

class ResMLPRegression(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            img_size: Union[Sequence[int], int],
            spatial_dims: int,
            num_classes=1, 
            num_clinical_features=10,
            dropout_rate=0.1,
            apply_tabular_embedding=False,
            tabular_embedding_size=96,
            num_embeddings_tabular = 32000, #default for llamaII
            conv1_t_size=7,
            conv1_t_stride=1,
            pretrained_vision_net = False,
            model_path = None, 
            freeze_encoder=False,
            act = "relu", #better not change this
            only_vision = False,
            only_clinical = False,
            num_hidden_layers_clinical = 2,
            num_hidden_layers_fusion = 4,
            ):
        super().__init__()
        
        self.only_vision = only_vision
        self.only_clinical = only_clinical
        self.multimodal = not only_vision and not only_clinical

        assert not (only_vision and only_clinical), "Only one of only_vision and only_clinical can be True. Set both to False in order to use multimodal model."

        #VISION
        num_vision_features = num_clinical_features if not apply_tabular_embedding else tabular_embedding_size #set to the same number of features
        if self.only_vision or self.multimodal:
            ResNetBlock = ResNetBlock_MONAI
            block_inplanes = [64, 128, 256, 512]
            self.encoder = ResNetMonai(
                block = ResNetBlock,
                layers = [3, 4, 6, 3],
                block_inplanes = block_inplanes,
                spatial_dims = spatial_dims,
                n_input_channels=in_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=False,
                shortcut_type="B",
                widen_factor= 1.0,
                num_classes = None, # #Attention here !!! not used, added a decoder on top
                feed_forward = False, # must be False !!!, due to pretraining, and decoder on top
                bias_downsample = True,  # for backwards compatibility (also see PR #5477)
            )
            if pretrained_vision_net and model_path is not None:
                #now load weights:
                print("Loading Weights from the Path {}".format(model_path))
                resnet_dict = torch.load(model_path, weights_only=True)
                resnet_weights = resnet_dict["state_dict"]

                # Remove items of resnet_weights if they are not in the resnet backbone (this is used in pretraining probably).
                # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
                # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
                # while pretraining with ResnetAutoEnc and are not a part of Resnet backbone.
                model_dict = self.encoder.state_dict()

                resnet_weights = {k: v for k, v in resnet_weights.items() if k in model_dict}
                model_dict.update(resnet_weights)
                self.encoder.load_state_dict(model_dict)
                del model_dict, resnet_weights, resnet_dict
                print("Pretrained Weights Succesfully Loaded !")
            else:
                print("Training from scratch.")

            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
                    #here also Avg pool will be frozen, which is no problem at all since it has no learnable parameters
                print("✓ encoder frozen")

            decoder = vision_head(block_inplanes[3] * ResNetBlock.expansion, num_vision_features)

            if act == "relu":
                self.vision_model = nn.Sequential(self.encoder, nn.Dropout(dropout_rate), decoder, nn.ReLU())
            else:
                self.vision_model = nn.Sequential(self.encoder, nn.Dropout(dropout_rate), decoder)

        #CLINICAL
        if self.only_clinical or self.multimodal:
            clinical_mlp_dim = 4*num_clinical_features if not apply_tabular_embedding else 4*tabular_embedding_size #times 4 is quite common
            if apply_tabular_embedding:
                TabularPreprocessing = TabularEmbedding(num_embeddings_tabular, tabular_embedding_size)
                if act == "relu":
                    self.clinical_model = nn.Sequential(TabularPreprocessing, MLPBlock(tabular_embedding_size, clinical_mlp_dim, dropout_rate), nn.ReLU())
                else:
                    self.clinical_model = nn.Sequential(TabularPreprocessing, MLPBlock(tabular_embedding_size, clinical_mlp_dim, dropout_rate))
            else:
                if act == "relu":
                    self.clinical_model = nn.Sequential(MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate), nn.ReLU())
                else:
                    self.clinical_model = MLPBlock(num_clinical_features, clinical_mlp_dim, dropout_rate)
        #update num_clinical_features to the new value
        num_clinical_features = tabular_embedding_size if apply_tabular_embedding else num_clinical_features
        
        #FUSION
        if self.multimodal:
            fusion_dim = num_clinical_features + num_vision_features #for concatenation of vision and clinical features
            fusion_mlp_dim = 4*fusion_dim
            if act == "relu":
                self.fusion = nn.Sequential(MLPBlock(fusion_dim, fusion_mlp_dim, dropout_rate), nn.ReLU())
            else:
                self.fusion = MLPBlock(fusion_dim, fusion_mlp_dim, dropout_rate)

        if self.multimodal:
            self.regression_head = Regression(fusion_dim, num_classes)
        elif self.only_vision:
            self.regression_head = Regression(num_vision_features, num_classes)
        elif self.only_clinical:
            self.regression_head = Regression(num_clinical_features, num_classes)
    
    def unfreeze_encoder_weights(self, num_layers_to_unfreeze=-1, block=4):
        """
        Unfreeze the weights of the encoder.
        If blocks_to_unfreeze is -1, unfreeze all weights.
        If blocks_to_unfreeze is a positive integer, unfreeze the last num_layers_to_unfreeze in block block.
        as ResNet34 has 4 blocks with  [3,4,6,3] layers, the layers are counted from end to beginning.
        self.encoder.layer4[-1].parameters() is the last layer of the last block.
        block = 1,2,3,4
        """
        assert block in [1,2,3,4], "block must be 1,2,3 or 4"
        if num_layers_to_unfreeze == -1:
            print("Unfreezing all weights of the encoder.")
            for p in self.encoder.parameters():
                p.requires_grad = True
        elif num_layers_to_unfreeze > 0:
            print(f"Unfreezing the last {num_layers_to_unfreeze} layers of block {block} of the encoder.")
            layer = self.encoder.layer4 if block == 4 else self.encoder.layer3 if block == 3 else self.encoder.layer2 if block == 2 else self.encoder.layer1
            for l in layer[-num_layers_to_unfreeze:]:
                for p in l.parameters():
                    p.requires_grad = True
        else:
            print("num_layers_to_unfreeze must be a positive integer or -1 to unfreeze all weights.")


    def forward(self, clinical_info, img):
        if self.multimodal:
            vision_features = self.vision_model(img) #shape is [batch_size, sequence_len, hidden_size_vision]
            tabular_features = self.clinical_model(clinical_info) #shape is [batch_size, num_features, clinical_mlp_dim]
            #transform the vision features to the same shape as the clinical features
            #first: flatten the vision features, was [1,sequence_len,hidden_size_vision] now [1,sequence_len*hidden_size_vision]
            
            #the following two line are not relavant anymore, as the cls token is used ant hence output dim already matches the clinical features
            #vision_features = vision_features.flatten(1)
            #vision_features = self.vision_transform(vision_features) #transform to the same shape as the clinical features with a linear layer (MLP)
            
            #concat the features
            mixed_features = torch.cat([vision_features, tabular_features], dim=1)
            x = self.fusion(mixed_features)
            x = self.regression_head(x)
            return x
        elif self.only_vision:
            vision_features = self.vision_model(img)
            x = self.regression_head(vision_features)
            return x
        elif self.only_clinical:
            tabular_features = self.clinical_model(clinical_info)
            x = self.regression_head(tabular_features)
            return x
