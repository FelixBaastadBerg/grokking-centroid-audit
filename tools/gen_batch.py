# 2b: nail the regime where Birdal/Andreeva trajectory PH-dim is MEASURABLE.
# Have full / 64 / 16. Fill the curve: 4, 32, 128, 512.
cfgs=[]
for bs in (4, 32, 128, 512):
    cfgs.append(f"--task add --p 23 --seed 0 --weight_decay 0.03 --train_fraction 0.8 --hidden 200 --num_epochs 20000 --save_every 200 --batch_size {bs} --dense_proj 128 --tag ht5b")
print("\n".join(cfgs))
