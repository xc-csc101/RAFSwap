from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, default='exp_celeba', help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default='checkpoints/iteration_200.pt', type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='data/images/', help='Path to directory of images to evaluate')
		self.parser.add_argument('--mask_path', type=str, default='data/label/', help='Path to directory of mask to evaluate')
		self.parser.add_argument('--txt_path', type=str, default='data/pair.txt', help='Path to directory of txt to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_false', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_false', help='Whether to resize outputs to 256x256 or keep at 1024x1024')

		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test/inference dataloader workers')

		# arguments for style-mixing script
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
		self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
		self.parser.add_argument('--latent_mask', type=str, default=None, help='Comma-separated list of latents to perform style-mixing with')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None,
		                         help='Downsampling factor for super-res (should be a single value for inference).')
		
		# arguments for transformer
		self.parser.add_argument('--depth', default=1, type=int, help='number of transformer layer')

	def parse(self):
		opts = self.parser.parse_args()
		return opts