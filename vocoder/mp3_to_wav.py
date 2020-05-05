import os
from joblib import Parallel, delayed
import multiprocessing
import subprocess

# idx = 0
def convert(file, root='/work/vlin2/accent_transfer/unbabel/e2e/'):
	subprocess.call(['ffmpeg -hide_banner -loglevel panic', '-i', '{}audio_samples/{}'.format(root, file),
                   '{}audio_wav/{}.wav'.format(root, file.split('.')[0])])

num_cores = multiprocessing.cpu_count() - 1
dirs = os.listdir('/work/vlin2/accent_transfer/unbabel/e2e/audio_samples')
Parallel(n_jobs=num_cores)(delayed(convert)(file) for file in dirs)
