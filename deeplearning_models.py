import torch.nn as nn
import torch
from dblur.models.nafnet import NAFNet
from torchmetrics.image import PeakSignalNoiseRatio
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class PSNRLoss(nn.Module):
    """
    PSNR loss for image deblurring used for training NAFNet. 
    
    Details regarding the loss function can be found in the paper "Simple 
    Baselines for Image Restoration".  
    """

    def __init__(self):
        super(PSNRLoss, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.psnr = PeakSignalNoiseRatio(data_range=1).to(device)
        self.mse_loss = nn.MSELoss()
        self.max_pixel_value = 255

    def forward(self, output, target):
        return -self.psnr(output, target)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Using VGG16 features for perceptual loss
        import torchvision.models as models

        vgg = models.vgg16(weights=None).features
        self.blocks = nn.ModuleList([
            vgg[:4],  # conv1_2
            vgg[4:9],  # conv2_2
            vgg[9:16],  # conv3_3
        ])

        # Register normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
            bl.eval()

    def normalize(self, x):
        return (x - self.mean) / self.std
        
    def forward(self, x, y):

        # Normalize both inputs
        x = self.normalize(x)
        y = self.normalize(y)
        
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss






class LitUModel_deblurring(pl.LightningModule):
    def __init__(self, learning_rate=7e-4, weight_decay=1e-4):
        super(LitUModel_deblurring, self).__init__()
        
        
        self.model = NAFNet(
            projected_in_channels=24,  # Reduced from 32
            enc_num=[1, 1, 1, 12],     # Reduced depth in the deepest layer
            middle_num=1,
            dec_num=[1, 1, 1, 1],      # Same decoder depth
            attn_expansion_factor=2,
            ffn_expansion_factor=2,
            gate_reduction_factor=2,
            dropout_rate=0,
            upscale_factor=2,
            attn_kernel_size=3,
            upscale_kernel_size=1,
            bias=True,
            upscale_bias=True
        )
        '''self.model = NAFNet(
            projected_in_channels=16,  # Reduced from 32
            enc_num=[1, 2, 4, 8],     # Reduced depth in the deepest layer
            middle_num=1,
            dec_num=[1, 1, 1, 1],      # Same decoder depth
            attn_expansion_factor=2,
            ffn_expansion_factor=2,
            gate_reduction_factor=2,
            dropout_rate=0.3,
            upscale_factor=2,
            attn_kernel_size=3,
            upscale_kernel_size=1,
            bias=True,
            upscale_bias=True)'''
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.psnr_loss = PSNRLoss()
        self.perceptual_loss = PerceptualLoss()
        
        # Loss weights
        self.lambda_mse = 0.0
        self.lambda_l1 = 0
        self.lambda_perceptual = 0.3
        self.lambda_psnr = 0.7

    def forward(self, x):
        return self.model(x)


    def compute_total_loss(self, outputs, targets):
        # MSE Loss
        mse = self.mse_loss(outputs, targets)
        
        # L1 Loss
        l1 = self.l1_loss(outputs, targets)

        psnr = self.psnr_loss(outputs, targets)
        
        # Perceptual Loss
        perceptual = self.perceptual_loss(outputs, targets)
        
        # Total weighted loss
        total_loss = (self.lambda_mse * mse + 
                     self.lambda_l1 * l1 + self.lambda_psnr * psnr
                     + self.lambda_perceptual * perceptual)
        
        return total_loss, mse, l1

    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        total_loss, mse, l1 = self.compute_total_loss(outputs, targets)
        
        # Log all losses
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_mae', l1, prog_bar=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        total_loss, mse, l1 = self.compute_total_loss(outputs, targets)
        
        # Log all losses
        self.log('val_total_loss', total_loss, prog_bar=True, on_epoch=True)
        self.log('val_mae', l1, prog_bar=True, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        return optimizer

    # Save function for the model state
    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}")

    # Class method to load the model for inference
    @classmethod
    def load_for_inference(cls, path, device='cpu'):
        # Create an instance without initializing other components
        model = cls()
        
        # Load state dict
        state_dict = torch.load(path, map_location=device,weights_only=True)
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"Model loaded from {path} for inference")
        return model
    

class LitUModel_dehazing(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, weight_decay=1e-5):
        super(LitUModel_dehazing, self).__init__()
        
        self.model = smp.Unet(
            encoder_name='mit_b3',
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation='sigmoid')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        
        # Loss weights
        self.lambda_mse = 0.2
        self.lambda_l1 = 0.4
        self.lambda_perceptual = 0.4

    def forward(self, x):
        return self.model(x)


    def compute_total_loss(self, outputs, targets):
        # MSE Loss
        mse = self.mse_loss(outputs, targets)
        
        # L1 Loss
        l1 = self.l1_loss(outputs, targets)
        
        # Perceptual Loss
        perceptual = self.perceptual_loss(outputs, targets)
        
        # Total weighted loss
        total_loss = (self.lambda_mse * mse + 
                     self.lambda_l1 * l1 + 
                     self.lambda_perceptual * perceptual)
        
        return total_loss, mse, l1, perceptual

    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        total_loss, mse, l1, perceptual = self.compute_total_loss(outputs, targets)
        
        # Log all losses
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_mae', l1, prog_bar=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        total_loss, mse, l1, perceptual = self.compute_total_loss(outputs, targets)
        
        # Log all losses
        self.log('val_total_loss', total_loss, prog_bar=True, on_epoch=True)
        self.log('val_l1', l1, prog_bar=True, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        return optimizer
    # Class method to load the model for inference
    @classmethod
    def load_for_inference(cls, path, device='cpu'):
        # Create an instance without initializing other components
        model = cls()
        
        # Load state dict
        state_dict = torch.load(path, map_location=device,weights_only=True)
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"Model loaded from {path} for inference")
        return model