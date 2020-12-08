"""Build vocabulary from manifest files.

Each item in vocabulary file is a character.
"""
import sys
import argparse
import functools
import codecs
import json
from collections import Counter
import os.path

def count_manifest(counter, manifest_path):
    with open(manifest_path) as f:
        for line in f:
            line_splits = line.strip().split()
            utt_id = line_splits[0]
            transcript = ''.join(line_splits[1:]) #以''作为分隔符，将ine_splits[1:]所有的元素合并成一个新的字符串
            for char in transcript:
                counter.update(char)#统计一个文件中，每个单词出现的次数


def main():
    #text = sys.argv[1]
    #count_threshold = int(sys.argv[2])
    #vocab_path = sys.argv[3]
    text = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/train/text"
    count_threshold = 0
    vocab_path = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/train/vocab"
    
    counter = Counter()#定义一个list数组，求数组中每个元素出现的次数
    count_manifest(counter, text)
    
    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True) # sorted() 函数对所有可迭代的对象进行排序操作 用x[1]这个数进行排序reverse = True 降序
    print (len(count_sorted))
    num = 1
    with open(vocab_path, 'w') as fout:
        fout.write('<unk> 1' + '\n')
        for char, count in count_sorted:
            if count < count_threshold: break
            num += 1
            fout.write(char + ' ' + str(num) + '\n')
    print (num)

if __name__ == '__main__':
    main()

