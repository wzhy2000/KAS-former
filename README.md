KAS-former
===============

KAS-former: a transformer-based model for predicting histone modifications using KAS-seq

Abstract:
--------
Histone modifications (HMs) play a critical role in
various biological processes, but annotating histone modifications
across different cell types using experimental methods alone is extremely challenging. Although many deep learning methods have
been developed to predict histone modifications, most rely solely
on DNA sequences and do not incorporate novel cell-specific
features. In this study, we propose KAS-former, a transformerbased model that integrates DNA sequences with cell-specific
features derived from KAS-seq data, enabling effective prediction
of histone modifications. Leveraging this transformer architecture coupled with dilated convolution, KAS-former achieves
a broad receptive field, effectively capturing cell type-specific
specificity from KAS-seq data. Our results demonstrate that
KAS-former achieves high accuracy in predicting histone modifications across multiple cell types and shows strong potential
for transcription factor prediction. By capturing cell-specific
features, this approach not only improves the accuracy of histone
modification predictions but also offers valuable insights into
the interplay between histone modifications and transcription
regulation.

1.Install KAS-former: 
==========================

Download
-----------------

KAS-former's source code is availiable on GitHub (https://github.com/wzhy2000/KAS-former). 

Required software
-----------------
* Python 3.9
* Tensorflow 2.13.0 (https://www.tensorflow.org/)
* pyBigwig (https://github.com/deeptools/pyBigWig)

2.Data preparation: 
==========================
Data download
-----------------
* **KAS-seq bigwig**: Download from the Gene Expression Omnibus (GEO) under accession number GSE139420 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139420).
* **hg19 Blacklist**: Obtain the hg19 blacklist regions from https://github.com/Boyle-Lab/Blacklist/.
* **Histone modification ChIP-seq**: The ChIP-seq data related can be available from the NCBI Gene Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo/) and the ENCODE project
* **ref_genome.fa**: Download the reference genome in FASTA format.  The hg19 reference genome can be found at the UCSC Genome.

Generate Dataset
-----------------

    python generate_dataset.py -b  hg19_blacklist.bed  -o ouput_path  -l seq_length  ref_genome.fa  target.txt  kas_id.txt 

    -o ouput_path      -- The path to save the output files.
    -l seq_length      -- [default=131072] Sequence length.
    ref_genome.fa      -- Reference genome (hg19).
    target.txt         -- The path of the Histone modifications bigWig files.
    kas_id.txt         -- The path of the KAS-seq bigWig file. This file should be in the same folder as the data.

Here is an example of target.txt, please replace the `file` with your own path for histone modification bigwig file:

| index | identifier | file                      | clip | sum_stat 
|-------|------------|---------------------------|------|----------
| 0     | H3K122ac   | ../data/H3K122ac.bw      | 384  | mean     
| 1     | H3K4me1    | ../data/H3K4me1.bw       | 384  | mean     
| 2     | H3K4me2    | ../data/H3K4me2.bw       | 384  | mean    
| 3     | H3K4me3    | ../data/H3K4me3.bw       | 384  | mean     
| 4     | H3K27ac    | ../data/H3K27ac.bw       | 384  | mean     
| 5     | H3K27me3   | ../data//H3K27me3.bw      | 384  | mean    
| 6     | H3K36me3   | ../data/H3K36me3.bw      | 384  | mean     
| 7     | H3K9ac     | ../data/w/H3K9ac.bw        | 384  | mean     
| 8     | H3K9me3    | ../data/local/w/H3K9me3.bw       | 384  | mean     
| 9     | H4K20me1   | ../data/H4K20me1.bw      | 384  | mean      

3.Train
===================
To train the model, simply run the following command:

    python train.py 

Before training the model, ensure that the dataset paths in `train.py` are correctly configured. This is crucial for the successful loading of the dataset during the training process.


4.Predict
===================

    python predict.py  -d output_predict_dataset  -o output_predict_results  -m model_path  --kas KAS-seq_bigwig --ref ref_genome.fa

    -d output_predict_dataset  -- The path to save the predict dataset files.
    -o output_path             -- The path to save the output bigwig files.
    -m model_path              -- The path to the trained model.
    --ref ref_genome.fa        -- Reference genome.(hg19 or mm10)
    --kas KAS-seq_bigwig       -- KAS-seq bigwig or bw file.

After the command execution, you will obtain 10 bigWig (bw) files for different histone modifications. You can visualize and inspect these files using tools such as IGV (Integrative Genomics Viewer) or Genome Browser. Alternatively, you can utilize the Python package pyBigWig to read the signal data.


**Notice:** 
That command takes more than 8 hours (depends on size of the train datasets) to execute on NVIDA 3090 GPU. Due to very long computational time, we don't suggest to run on CPU nodes.




