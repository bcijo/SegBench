# SegBench
## This repository will be focused towards benchmarking all the popular segmentation models against standard open-source datasets.

## List of Models

### U-Net (2015)
- Key innovation: Symmetric encoder-decoder architecture
- Architecture: Contracting path + expanding path with skip connections
- Impact: Still widely used, especially in medical imaging
- Notable feature: Copy and crop feature maps from encoder to decoder
- Paper: https://arxiv.org/pdf/1505.04597 
- Code: https://www.kaggle.com/code/bryanb/image-segmentation-u-net

SegNet (2015)
Key innovation: Encoder-decoder with pooling indices
Architecture: VGG16-based encoder with corresponding decoder
Notable feature: Memory efficient upsampling using pooling indices
Paper: https://arxiv.org/pdf/1511.00561
Code: https://www.kaggle.com/code/harlequeen/semantic-segmentation-of-plants-with-segnet
PSPNet (2016)
Full name: Pyramid Scene Parsing Network
Key innovation: Pyramid Pooling Module
Architecture: ResNet backbone with hierarchical global prior
Notable feature: Multi-scale feature aggregation
Paper: https://arxiv.org/pdf/1612.01105
Code: https://www.kaggle.com/code/michaelcripman/image-segmentation-using-pspnet?scriptVersionId=61005697
DeepLab v3+ (2018)
Key innovation: Atrous Spatial Pyramid Pooling (ASPP)
Architecture: Encoder-decoder with ASPP and modified Xception backbone
Notable feature: Multi-scale feature extraction with dilated convolutions
Paper: https://arxiv.org/pdf/1802.02611
Code: 
HRNet (2019)
Full name: High-Resolution Network
Key innovation: Parallel high-to-low resolution convolutions
Architecture: Maintains high-resolution representations throughout
Notable feature: Multi-scale fusion across all stages
Paper: 
Code: 
SETR (2021)
Full name: SEgmentation TRansformer
Key innovation: Pure transformer-based segmentation
Architecture: ViT-based encoder with progressive upsampling decoder
Notable feature: First successful application of transformers to segmentation
Paper: 
Code: 
Swin Transformer - 2021
Key Idea: Introduced a hierarchical transformer model with shift-based windows, enabling it to capture both local and global dependencies.
Paper:
Code: 
Segment Anything Model (SAM) (2023)
Key innovation: Prompt-based segmentation
Architecture: Image encoder + prompt encoder + mask decoder
Notable feature: Zero-shot segmentation capabilities
Impact: Changed paradigm from class-specific to prompt-based segmentation
Paper: 
Code: 
