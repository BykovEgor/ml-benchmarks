import logging

import torch.utils.data as tud

import data_loader

logging.basicConfig(level=logging.INFO)


def seek_max_batch_size(dataset, data_processors, function):
    logging.info(" SEEKING MAX BATCH SIZE POSSIBLE:")
    batch_size = 1
    while True:
        print("\tTrying batch size: {}".format(batch_size), end="\t")
        dataloader = tud.DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=data_processors,
                                    pin_memory=True,
                                    prefetch_factor=batch_size,
                                    collate_fn=data_loader.bertBatchCollate
                                    )
        
        try:
            for i_batch, batch in enumerate(dataloader):
                if i_batch > 0:
                    break
                ## We must take result from GPU
                _ = function(batch)
            print("SUCCESS!")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("FAILED!")
                return batch_size - 1
            else:
                raise e
        
        batch_size += 1
