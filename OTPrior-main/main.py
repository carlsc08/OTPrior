import subprocess


config = 'configs/64d_cosine.yaml'
script1 = 'train/prior.py'
script2 = 'train/ae.py'

print(f"Running: {script1}")
#subprocess.run(['python', script1, config])
print(f"Running: {script2}")
subprocess.run(['python', script2, config])