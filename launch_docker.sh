docker run -it -d --shm-size 10g --net=host --ipc=host --privileged \
  --name vllm-gpt-oss-int4 \
  -v /mnt/model_cache/:/root/.cache/ \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v /home/dbyoung/Workspace:/work \
  --device /dev/dri:/dev/dri \
  --entrypoint=  gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:multi-bmg_release_ww31.5  /bin/bash