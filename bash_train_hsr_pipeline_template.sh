LOGDIR="./records/pipeline/model_logs" 
if [ ! -d "$LOGDIR" ]; then
  mkdir -p $LOGDIR
fi

# hsrnet network archietecture
nb_kernels=30 # other parameters [30,48,60,72,90]
out_channels=10

# network type, hsr
net=hsr # hsr_context
hsr_compress=sep # 3 parameters: None, "sep", "bsep" for HSRNet, SepHSR, and BSepHSR.

if [ $hsr_compress = None ]; then
  net_name=hsr
elif [ $hsr_compress = sep ]; then
    net_name=sephsr
elif [ $hsr_compress = bsep ]; then
    net_name=bsephsr
fi

echo "Start training HSRNet with imdb dataset!"

db=imdb # dataset: imdb, wiki, morph2
batch_size=128
nb_epochs=160

python train_hsrfamily.py --db $db --pipeline 1 --batch_size $batch_size --nb_epochs $nb_epochs --net $net --nb_kernels $nb_kernels --hsr_compress $hsr_compress --out_channels $out_channels |& tee $LOGDIR/${net_name}${nb_kernels}-${out_channels}_${db}_logs.txt

db=wiki
batch_size=50
nb_epochs=160

echo "Start training HSRNet with wiki dataset!"
python train_hsrfamily.py --db $db --pipeline 1 --batch_size $batch_size --nb_epochs $nb_epochs --net $net --nb_kernels $nb_kernels --hsr_compress $hsr_compress --out_channels $out_channels |& tee $LOGDIR/${net_name}${nb_kernels}-${out_channels}_${db}_logs.txt

db=morph2
batch_size=50
nb_epochs=160

echo "Start training HSRNet with morph2 dataset!"
python train_hsrfamily.py --db $db --pipeline 1 --batch_size $batch_size --nb_epochs $nb_epochs --net $net --nb_kernels $nb_kernels --hsr_compress $hsr_compress --out_channels $out_channels |& tee $LOGDIR/${net_name}${nb_kernels}-${out_channels}_${db}_logs.txt