from shutil import copyfile
import glob, os

for img_dir in glob.glob('data/101_ObjectCategories/*'):
	base_dir = os.path.basename(img_dir)

	os.makedirs('data/train_caltech101/' + base_dir, exist_ok=True)
	os.makedirs('data/val_caltech101/' + base_dir, exist_ok=True)

	count = 0
	for img_path in glob.glob(img_dir + '/*'):
		print(img_path)
		# save to train
		if count < 10:
			copyfile(img_path, 'data/train_caltech101/' + base_dir + '/' + os.path.basename(img_path))
		else:
			copyfile(img_path, 'data/val_caltech101/' + base_dir + '/' + os.path.basename(img_path))
		count += 1
