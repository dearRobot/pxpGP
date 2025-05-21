# gaussian_processes

Run  cGP using c-ADMM optimization in multi-agent environment:

```
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=12345 cgp_train.py
```

Or
```
python3 -m torch.distributed.launch --nproc_per_node=2 --master_addr=localhost --master_port=12345 cgp_train.py
```

Where,

1. `nproc_per_node` : No of agent in system
2. `master_addr` :  IP address of central node
3. `master_port` : Port ID of central node
