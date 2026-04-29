from src.worlds.micro_world.world import create_micro_world
from src.worlds.healthcare_world.world import create_healthcare_world
from src.worlds.ecommerce_world.world import create_ecommerce_world
from src.worlds.cloud_devops_world.world import create_cloud_devops_world
from src.worlds.social_media_world.world import create_social_media_world

WORLD_FACTORIES = {
    'micro_world': create_micro_world,
    'healthcare_world': create_healthcare_world,
    'ecommerce_world': create_ecommerce_world,
    'cloud_devops_world': create_cloud_devops_world,
    'social_media_world': create_social_media_world,
}
