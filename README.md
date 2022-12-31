# README

## Environment:

```
python=3.8
'pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117' // !!!
pillow==9.2.0
pandas==1.5.2
visdom==0.2.3
matplotlib==3.6.2
visdom==0.2.3
dlib==19.24.0
opencv-python==4.6.0.66
```

Noted: I trained the model on Windows with a nvidia GPU, so pytorch-cuda=11.7 is needed.
If your Windows have no GPU device, use 'pip3 install torch torchvision torchaudio' instead.
If Mac, use 'pip3 install torch torchvision torchaudio'.
And 'pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu' for Linux.
**See https://pytorch.org/get-started/locally/ for more information.**

## How to run the project:

```shell
python main.py
```

After execute the code, four models will start training, and after that the models will be tested one by one. You're supposed to close some matplotlib windows when executing to keep it going run.
