DATA_DIR=/home/yang/link/data
QUERY_MODEL=sysu_thermal_resnet50_drop_0.5_lr_0.01_hid_512_sgd_20190812-212745
GALLERY_MODEL=sysu_visible_resnet50_drop_0.0_lr_0.01_hid_512_sgd_20190813-165716
CROSS_MODEL=sysu_cross_resnet50_drop_0.0_lr_0.01_hid_512_sgd_20190814-220610

.PHONY: preprocess
preprocess:
	python preprocess.py \
		--data_dir $(DATA_DIR)/SYSU-MM01

.PHONY: train_cross
train_cross:
	python train_cross.py \
		--dataset sysu \
		--data_dir $(DATA_DIR)/SYSU-MM01 \
		--ckpt_dir $(DATA_DIR)/ckpts \
		--lr 0.01 \
		--lr_steps 40 60 \
		--batch_train 64 \
		--batch_test 512 \
		--margin 0.6

.PHONY: train
train:
	python train.py \
		--dataset sysu \
		--data_dir $(DATA_DIR)/SYSU-MM01 \
		--ckpt_dir $(DATA_DIR)/ckpts \
		--lr_steps 20 30 \
		--modal visible \
		--batch_train 64 \
		--batch_test 512

.PHONY: test
test:
	python test.py \
		--data_dir $(DATA_DIR)/SYSU-MM01 \
		--batch_size 512 \
		--cross_model_path $(DATA_DIR)/ckpts/$(CROSS_MODEL).pth \
		--query_model_path $(DATA_DIR)/ckpts/$(QUERY_MODEL).pth \
		--gallery_model_path $(DATA_DIR)/ckpts/$(GALLERY_MODEL).pth
