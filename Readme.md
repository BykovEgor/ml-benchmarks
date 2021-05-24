## Basic Usage

### Requirements 
 - Docker
 - Python

### Setting Environment (Only for local launches)

1 - Run `pip install -r requirements.txt`

### Mock data

1 - Fill `benchmark/mock_data.txt` with samples to process.   
2 - Run `benchmark/mock_data_generator.py` to generate mock data files.  

### Running Locally

1 - Run `benchmark/inference.py --$RUN_PARAMS`
  
  
  
### Running In Docker

#### Building docker images

1 - Run `benchmark/build_docker_images.sh` to build all images 

#### Running benchmarks

1 - Run `docker run $DOCKER_IMG_NAME --$RUN_PARAMS`