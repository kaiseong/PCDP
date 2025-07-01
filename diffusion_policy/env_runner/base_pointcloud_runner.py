from typing import Dict
from diffusion_policy.policy.base_pointcloud_policy import BasePointCloudPolicy

class BasePointCloudRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePointCloudPolicy) -> Dict:
        raise NotImplementedError()
