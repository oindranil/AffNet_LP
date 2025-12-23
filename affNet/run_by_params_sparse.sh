python predict_link_sparse.py --dataset=Photo --emb_features=186 --n_heads=4 --max_nodes=4000 --init_lr=0.002 --epochs=2000
python predict_link_sparse.py --dataset=ogbl-collab --emb_features=80 --n_heads=4 --max_nodes=4000 --init_lr=0.001 --epochs=1000
python predict_link_sparse.py --dataset=ogbl-citation2 --emb_features=128 --n_heads=4 --max_nodes=4000 --init_lr=0.0001 --epochs=2000
