from src.models.lr_backend import MultiscaleMultibranchTCN
from src.models.lr_frontend import  VisualFrontend, LandmarksFrontend, FusionFrontend
from src.models.lipreading import Lipreading

def model_factory(cfg):

    visual_frontend = None
    landmarks_frontend = None
    fusion_frontend = None

    if "video" in cfg.frontend:
        visual_frontend = VisualFrontend(cfg.frontend.video)
    if "landmarks" in cfg.frontend:
        landmarks_frontend = LandmarksFrontend(cfg.frontend.landmarks)
    if "fusion" in cfg.frontend and "video" in cfg.frontend and "landmarks" in cfg.frontend:
            fusion_frontend = FusionFrontend(cfg.frontend.fusion)

    # backend
    if cfg.backend.type == 'tcn':
        backend = MultiscaleMultibranchTCN(cfg.backend)
    else:
        raise ValueError(f"Model {cfg.backend} backend not recognized")
        return None

    model = Lipreading(visual_frontend, landmarks_frontend, fusion_frontend, backend)
    return model



