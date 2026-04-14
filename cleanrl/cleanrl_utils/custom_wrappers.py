from typing import Literal

import numpy as np
import torch
import gymnasium as gym

class DINOv2FeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, model: Literal['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'] = 'dinov2_vits14_reg'):
        super().__init__(env)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dino_model = torch.hub.load('facebookresearch/dinov2', model)
        self.dino_model.eval()
        self.dino_model.to(self.device)

        # parent observation should be images
        if not isinstance(env.observation_space, gym.spaces.box.Box) or len(env.observation_space.shape) not in [3, 4]:
            raise ValueError(f"DINOv2FeatureWrapper only works with image observations! Got {env.observation_space}")
        
        if len(env.observation_space.shape) == 4:
            new_obs_shape = (env.observation_space.shape[0], self.dino_model.embed_dim)
        else:
            new_obs_shape = (self.dino_model.embed_dim,)

        self.observation_space = gym.spaces.Box(
            low=-np.float32('inf'),
            high=np.float32('inf'),
            shape=new_obs_shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        assert len(observation.shape) in [3, 4], f"Only supports single image or batch of images, got {observation.shape}"
        
        observation = torch.tensor(observation, device=self.device).float()
        has_batch_dim = True
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)  # add batch dimension
            has_batch_dim = False

        # resize
        if observation.shape[-1] == 3:
            observation = observation.permute(0, 3, 1, 2)  # NHWC to NCHW
        observation = torch.nn.functional.interpolate(observation, size=(224, 224), mode='bilinear', align_corners=False)
        
        # normalize
        if observation.max() <= 1.0:
            observation = observation * 255
        observation = (observation - torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)) / torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)

        with torch.no_grad():
            features = self.dino_model(observation)
            
        if not has_batch_dim:
            features = features.squeeze(0)

        return features.cpu().numpy()

class RenderObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(600, 400, 3),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return self.env.render()

class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(240, 160, 1),
            dtype=np.float32
        )

    def observation(self, obs):
        dims = len(obs.shape)
        obs = obs / 255.0 * 2 - 1
        obs = cv2.resize(obs, (160, 240), interpolation=cv2.INTER_AREA)
        if dims == 3 and len(obs.shape) == 2:
            obs = obs[..., np.newaxis]        
        return obs