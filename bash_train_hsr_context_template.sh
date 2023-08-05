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

db=morph2_context
batch_size=50
nb_epochs=600
net_name=${net_name}_context

echo "Start training HSRNetContext with morph2_context dataset!"
python train_hsrfamily.py --db $db --pipeline 1 --batch_size $batch_size --nb_epochs $nb_epochs --net $net --nb_kernels $nb_kernels --hsr_compress $hsr_compress --out_channels $out_channels |& tee $LOGDIR/${net_name}${nb_kernels}-${out_channels}_${db}_logs.txt