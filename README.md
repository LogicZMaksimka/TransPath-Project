# Checkpoint 2 

## Maps generation

We adopted [TMP](https://arxiv.org/abs/2009.07476) (Tiled Motion Planning) dataset [generation script](https://github.com/omron-sinicx/planning-datasets/blob/icml2021/1_TiledMP.sh) add modified hyperparams as showed below:


```
python generate_spp_instances.py 
    --input-path "data/mpd/original/*" 
    --output-path data/mpd/ 
    --maze-size 128 
    --mechanism moore 
    --edge-ratio 0.25 
    --tile-size=2 
    --train-size 800 
    --valid-size 100 
    --test-size 100
```

## Tasks generation

Task generation script
```
python task_generation.py --maps_path <path_to_maps>
```

## Model training

View FS+PPM model **training report**:  
https://api.wandb.ai/links/blain/4679lk9x  


## Model evaluation
Detailed evaluation of FS+PPM model:   
https://github.com/LogicZMaksimka/TransPath-Project/blob/master/visualise_metrics.ipynb
