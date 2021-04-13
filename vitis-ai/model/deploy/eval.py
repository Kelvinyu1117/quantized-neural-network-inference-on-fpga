#!/usr/bin/python3
from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import math
divider='---------------------------'

def load_test_images(path):
    import pickle as pk
    
    with open(path, 'rb') as f:
        imgs = pk.load(f)

    return imgs
            

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize(imgs):
    import matplotlib.pyplot as plt
    X = imgs['data']
    Y = imgs['labels']
    label_names = load_label_names()
    
    fig, axes1 = plt.subplots(3,3,figsize=(10,10))
    for j in range(3):
        for k in range(3):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].set_title(label_names[Y[i:i+1][0]])
            axes1[j][k].imshow(X[i:i+1][0])
            
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = outputData[0][j]
            write_index += 1
        count = count + runSize


def app(img, labels, threads, model):

    runTotal = len(img)
    img = [np.transpose(i, [1, 2, 0]) for i in img]

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    ''' preprocess images '''
    print('Pre-processing',runTotal,'images...')
    

    '''run threads '''
    print('Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(img)
        else:
            end = start+(len(img)//threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, runTotal, timetotal))

    


    ''' post-processing '''
    correct = 0
    wrong = 0

    import pickle as pk
    
    with open(f'inference_result_{model}-thread{threads}.pkl', 'wb') as f:
        pk.dump(out_q, f)

    for i in range(len(out_q)):
      
        prediction = out_q[i]
        ground_truth = labels[i][0]
        if (ground_truth==np.argmax(prediction)):
            correct += 1
        else:
            wrong += 1
    accuracy = correct/len(out_q)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' %(correct,wrong,accuracy))
    print(divider)

    return

def main():
    ap = argparse.ArgumentParser()
    IMG_PATH = './data/test_imgs'
    imgs = load_test_images(IMG_PATH)


    ap.add_argument('-t', '--threads',   type=str, default="1",        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',     type=str, default='CifarResNet_int_u50.xmodel', help='Path of xmodel. Default is CifarResNet_int_u50.xmodel')
    args = ap.parse_args() 
    print(f"# of threads: {args.threads}")
    print(f"model: {args.model}")
    app(imgs['data'], imgs['labels'], int(args.threads),args.model)

if __name__ == '__main__':
  main()