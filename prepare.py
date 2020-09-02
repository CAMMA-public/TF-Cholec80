'''
Author: Tong Yu
Copyright (c) University of Strasbourg. All Rights Reserved.
'''

import json
import argparse
import requests
import hashlib
import tarfile
import os


URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz"
CHUNK_SIZE = 2 ** 20

parser = argparse.ArgumentParser()
parser.add_argument("--data_rootdir")
parser.add_argument("--verify_checksum", action="store_true")
parser.add_argument("--keep_archive", action="store_true")
args = parser.parse_args()


outfile = os.path.join(args.data_rootdir, "cholec80.tar.gz")
outdir = os.path.join(args.data_rootdir, "cholec80")

print("Downloading archive to {}".format(outfile))
with requests.get(URL, stream=True) as r:
  r.raise_for_status()
  with open(outfile, "wb") as f:
    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
      f.write(chunk)

if args.verify_checksum:
  print("Verifying checksum")
  m = hashlib.md5()
  with open(outfile, 'rb') as f:
    while True:
      data = f.read(CHUNK_SIZE)
      if not data:
        break
      m.update(data)
  chk = m.hexdigest()
  with open("checksum.txt") as f:
    true_chk = f.read()
  print("Checksum: {}".format(chk))
  assert(m.hexdigest() == chk)

print("Extracting files to {}".format(outdir))
with tarfile.open(outfile, "r") as t:
  t.extractall(outdir)

if not args.keep_archive:
  os.remove(outfile)

with open("config.json", "r") as f:
  config = json.loads(f.read())

config["cholec80_dir"] = os.path.join(outdir, "cholec80")
json_string = json.dumps(config, indent=2, sort_keys=True)

with open("config.json", "w") as f:
  f.write(json_string)

print("All done - config saved to {}".format(
  os.path.join(os.getcwd(), "config.json"))
)
