from diffusion_policy.policy.base_pointcloud_policy import BasePointCloudPolicy
from diffusion_policy.env_runner.base_pointcloud_runner import BasePointCloudRunner

class RealStackPointCloudRunner(BasePointCloudRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BasePointCloudPolicy):
        return dict()