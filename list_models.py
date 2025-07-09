import os
import torchsig

models_path = os.path.join(torchsig.__path__[0], "models")

print("Contents of torchsig.models:")
for root, dirs, files in os.walk(models_path):
    for name in dirs + files:
        print(os.path.relpath(os.path.join(root, name), models_path))
