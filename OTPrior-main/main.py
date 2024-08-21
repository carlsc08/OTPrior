import subprocess


config = 'configs/2d_cosine_mnist.yaml'
script1 = 'train/prior.py'
script2 = 'train/ae.py'

print(f"Running: {script1}")
subprocess.run(['python', script1, config])
print(f"Running: {script2}")
subprocess.run(['python', script2, config])