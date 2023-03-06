from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_swap': {
		'transforms': transforms_config.SwapTransforms,
		'train_image_root': dataset_paths['celeba_train_img'],
		'train_mask_root': dataset_paths['celeba_train_mask'],
		'test_image_root': dataset_paths['celeba_test_img'],
		'test_mask_root': dataset_paths['celeba_test_mask'],
	}
}
