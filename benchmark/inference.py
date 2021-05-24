import argparse
import json
import logging
import os
import subprocess
import timeit

import torch
import torch.utils.data as tud
from transformers import BertModel

import data_loader
import utils

logging.basicConfig(level=logging.INFO)


def transform(batch):
    for key in batch:
        batch[key] = batch[key].to(device, non_blocking=True)
    
    output = model(**batch).last_hidden_state
    
    return output


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',
                        type=int,
                        default=0,
                        help="Debug regime. Add number of batches to process")
    
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help="Number of documents in a batch. Defaul: 1")
    
    parser.add_argument('--data-processors',
                        type=int,
                        default=1,
                        help="The number of processes to shuffle the data. Defaul: 1")
    
    parser.add_argument('--data-folder',
                        type=str,
                        default="./data/",
                        help="Folder containg data. Defaul: ./data")
    
    parser.add_argument('--cpu-only',
                        action='store_true',
                        default=False,
                        help="Flag to tern off CUDA computation. Defaul: false")
    
    parser.add_argument('--seek-batch-size',
                        action='store_true',
                        default=False,
                        help="Find maximum batch size. Defaul: false")
    
    args = parser.parse_args()
    
    gpu_flag = not args.cpu_only and torch.cuda.is_available()
    
    if gpu_flag:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    model.to(device)
    
    logging.info("\n###########################################################\n")
    p = subprocess.Popen(["cat", "/proc/cpuinfo"], stdout=subprocess.PIPE)
    cpuinfo, error = p.communicate()
    print("\nCPU INFO:\n", str(cpuinfo, encoding="UTF-8"))
    print("\nPyTorch version: {}".format(torch.__version__))
    
    if gpu_flag:
        
        print("PyTorch cuda compiled version: {}".format(torch._C._cuda_getCompiledVersion()))
        print("\nCUDA is available. Following devices found:")
        for i in range(torch.cuda.device_count()):
            print("Device: {}".format(i))
            print("\tName: {}".format(torch.cuda.get_device_name(i)))
            print("\tProperties: {}".format(torch.cuda.get_device_properties(i)))
    
    print("\nData folder: {}".format(args.data_folder))
    print("Data processors: {}".format(args.data_processors))
    
    dataset = data_loader.ArticlesDataset(args.data_folder, "file_*.txt", transform=True)
    if args.seek_batch_size:
        print("\n\t {}".format(json.dumps(
            {"max_batch_size": utils.seek_max_batch_size(dataset, args.data_processors, transform)})))
        
        metrics = {
            "runtime": 0,
            "docs_per_sec": 0,
            "secs_per_doc": 0,
            "Kb_per_sec": 0,
            "args": args.__dict__
        }
        print("INFERENCE METRICS:\n\t{}\n".format(json.dumps(metrics)))
        
        exit()
    
    print("Batch size: {}".format(args.batch_size))
    
    dataloader = tud.DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.data_processors,
                                pin_memory=True,
                                prefetch_factor=args.batch_size,
                                collate_fn=data_loader.bertBatchCollate
                                )
    
    starttime = timeit.default_timer()
    
    total_batches = len(dataloader)
    
    # Cut data size if --dry-run mode:
    if args.dry_run != 0:
        print("\n Dry Run mode. Only {} batches will be processed.".format(args.dry_run))
        total_batches = min(total_batches, args.dry_run)
    
    print("Total Batches: {}".format(total_batches))
    
    total_files = len(dataset)
    ten_persent = int(total_batches / 10)
    file_size_Kb = os.stat(dataset.files[0]).st_size / 1024
    mem_summary = False
    
    print("\n Processing Data:")
    for i_batch, batch in enumerate(dataloader):
        if i_batch % ten_persent == 0:
            print("\t{:.1f}% of files processed".format(i_batch * args.batch_size * 100 / total_files))
            # if gpu_flag:
            #     # print("\t\t", "VRAM reserved: ", torch.cuda.memory_reserved())
            #     print(torch.cuda.memory_stats_as_nested_dict()['active_bytes']['all']['current'])
            #     print(torch.cuda.memory_stats_as_nested_dict()['inactive_split_bytes']['all']['current'])
            #     print(torch.cuda.memory_snapshot())
        
        _ = transform(batch)
        
        if (args.dry_run != 0) and (i_batch > args.dry_run - 1):
            break
    
    print("\t100% of files processed\n")
    
    runtime = timeit.default_timer() - starttime
    
    metrics = {
        "runtime": runtime,
        "docs_per_sec": total_files / runtime,
        "secs_per_doc": runtime / total_files,
        "Kb_per_sec": (file_size_Kb * total_files) / runtime,
        "args": args.__dict__
    }
    
    print("INFERENCE METRICS:\n\t{}\n".format(json.dumps(metrics)))
    print("\n###########################################################\n")
