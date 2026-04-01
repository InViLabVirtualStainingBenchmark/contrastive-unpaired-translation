#Changes introduced through forking the main model

## Commands used to run the models

**IMPORTANT NOTE**
Despite the model claiming to use Python 3.6, the actual version needed to create a correct environment is
Python 3.9 - the older version is not compatible with modern hardware and will cause the training to fail, 
resuling in extreme noise and lack of loss values. 


1. **Create the environment**: conda create -n cut-py39 python=3.9
2. **Activate the environemnt**: conda activate cut-py39
3. **Install the dependencies**: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
                                pip install dominate packaging GPUtil
                                pip install visdom --no-build-isolation
4. **Download the dataset**: bash ./datasets/download_cut_dataset.sh grumpifycat
5. **Train**: python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --n_epochs 1 --n_epochs_decay 1 --save_epoch_freq 1 --save_latest_freq 100 --display_id 0
6. **Test**: python test.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --epoch latest --phase train

*Note*: for full training remove --n_epochs 1 --n_epochs_decay 1 --save_latest_freq 100 and let it run with the defaults (200+200 epochs).