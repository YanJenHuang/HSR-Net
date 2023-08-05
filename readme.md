# Download datasets

- Download imdb and wiki "face only" datasets in:
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- md5sum
-- 44b7548f288c14397cb7a7bab35ebe14  imdb_crop.tar
-- f536eb7f5eae229ae8f286184364b42b  wiki_crop.tar

- Download morph2 (MORPH Academic Set) datasets:
https://uncw.edu/myuncw/research/innovation-commercialization/technology-portfolio/morph

# Create datasets

- create imdb datasets and save as imdb.npz file in ./datasets folder
``` python create_datasets/create_imdbwiki.py --db imdb --output datasets/imdb.npz ```
- create wiki datasets and save as wiki.npz file in ./datasets folder
``` python create_datasets/create_imdbwiki.py --db imdb --output datasets/wiki.npz ```
- create morph2 datasets and save as morph2.npz file in ./datasets folder
``` python create_datasets/create_morph.py --output datasets/morph2.npz ```

- create mutli resolution Morph2 dataset.
``` python create_datasets/create_morph-three_resolution.py --output datasets/morph2_context.npz ```

# Train HSRNet
Using the bash script 'bash_train_hsr_template.sh' to train from scratch and record the logs.
This template script trains SepHSR(30,10) with morph2 dataset from scratch with 50 batchsize and 160 epochs and records the log in './records/model_logs' folder.

The arguments in the bash script can be modified:

- nb_kernels = 30 (*integer*)
- out_channels = 10 (*integer*)
- hsr_compress = sep (*string*),  other valid value: None, sep, bsep
- db = morph2 (*string*), other valid value: imdb, wiki, morph2
- batch_size = 50 (*integer*), [imdb: 128, wiki: 50, morph2: 50] for our experiments.
- nb_epochs = 160 (*integer*)

```console
$ bash bash_train_hsr_template.sh
```

# Train HSRNet with IMDB > WIKI > MORPH2 (pipeline)
Using the bash script 'bash_train_hsr_pipeline_template.sh' to first train with imdb, wiki, and then morph2.

```console
$ bash bash_train_hsr_pipeline_template.sh
```

# Train HSRNetContext with morph2_context (3-resolutions)
This requires pipeline model's checkpoint after training the wiki dataset.
```console
$ bash bash_train_hsr_context_template.sh
```