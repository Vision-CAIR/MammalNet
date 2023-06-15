#!/bin/bash


#python build_rawframes.py  /home/chenj0g/dummy_videos/ /home/chenj0g/rawframes_test --level 1 --flow-type tvl1 --ext mp4 --task both  --new-short 320 --num-gpu 4 --num-worker 64

python build_rawframes.py  /home/chenj0g/dummy_videos/ /home/chenj0g/rawframes_test --level 1 --flow-type tvl1 --ext mp4 --task rgb  --new-short 320 --num-gpu 4 --num-worker 64 --use-opencv
