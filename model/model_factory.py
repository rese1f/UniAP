import os

from dataset.UniASET_constants import TASKS, TASKS_GROUP_DICT

from .beit.beit import BEiTEncoder, load_beit_ckpt
from .dpt.dpt import DPT
from .UniAP import UniAP, UniAPImageBackbone, UniAPLabelBackbone, UniAPMatchingModule


def get_model(config, device=None, verbose=True, load_pretrained=True):
    image_backbone = create_image_backbone(config, verbose=verbose, load_pretrained=load_pretrained)
    label_backbone = create_label_backbone(config)
    
    dim_w = image_backbone.dim_hidden
    dim_z = label_backbone.dim_hidden

    image_backbone = UniAPImageBackbone(image_backbone)
    label_backbone = UniAPLabelBackbone(label_backbone)
    matching_module = UniAPMatchingModule(dim_w, dim_z, config)

    model = UniAP(image_backbone, label_backbone, matching_module)
        
    if device is not None:
        model = model.to(device)

    return model


def create_image_backbone(config, verbose=True, load_pretrained=True):
    if config.stage == 0:
        n_tasks = len(TASKS)
    else:
        n_tasks = len(TASKS_GROUP_DICT[config.task])
    
    backbone = BEiTEncoder(
        config.image_backbone,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        attn_drop_rate=config.attn_drop_rate,
        n_tasks=n_tasks,
        n_levels=config.n_levels,
        bitfit=config.bitfit,
    )
    backbone.dim_hidden = backbone.embed_dim

    if load_pretrained and config.image_encoder_weights == 'imagenet':
        ckpt_path = os.path.join('model/pretrained_checkpoints',
                                 f'{config.image_backbone.replace("in22k", "pt22k")}.pth')

        if getattr(config, 'bitfit', False):
            n_bitfit_tasks = n_tasks
        else:
            n_bitfit_tasks = 0

        load_beit_ckpt(backbone.beit, ckpt_path, n_bitfit_tasks=n_bitfit_tasks, verbose=verbose)
                
    return backbone


def create_label_backbone(config):
    backbone = DPT(config.label_backbone)
    backbone.dim_hidden = backbone.embed_dim
        
    return backbone