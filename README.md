# Primepore
An analytical tool for detecting RNA modifications, capable of detecting three types of modifications and outputting modification predictions at the single-molecule level.
# introduction
Primepore is a high-performance software tool designed to detect and quantify three RNA modifications—m6A, m5C, and inosine—from Nanopore Direct RNA Sequencing data. The workflow consists of four stages, delivering per-site probabilities, modification proportions, and single-molecule modification status.  
![alt](Primepore.svg)
## Key Features
* Simultaneous detection of three RNA modifications (m6A, m5C, inosine)
* End-to-end four-stage workflow: Preprocessing, Classification (Transformer), Regression, Clustering
* Transformer-based classifier for initial screening of modification signals
* Regression stage to predict modification proportions
* Clustering stage to output per-molecule modification states
* Integration with traditional signal and alignment tools (basecalling, mapping, nanopolish) and Groundtruth alignment for feature extraction
* Outputs suitable for downstream statistics, visualization, and comparative analyses
## Workflow
### 1. Preprocessing and Raw Data Alignment
* Basecalling, read mapping, and event alignment
* Groundtruth alignment to reference
* Feature extraction and data integration to prepare inputs for modeling
### 2. Classification Stage
* Transformer-based network for preliminary screening of modification signals
* Distinguishes among m6A, m5C, inosine, and unmodified signals using sequence and signal context
### 3. Regression Stage
* Predicts modification proportions at identified sites
* Outputs continuous proportion estimates with associated uncertainty
### 4. Clustering Stage
* Outputs per-molecule modification status based on predicted proportions
* Groups reads by modification profiles to reveal heterogeneity and epitranscriptome structure  
![alt](workflow.svg)
# Requisites
## Data preparing:
| Data | Note |
| :------------ |:---------------:|
| fast5 files |containing raw current signals|
| reference.fa      | genome.fa (hg19.fa or hg38.fa) |
| methylation_rate.csv      | methylation-rate groundtruth, needed only for training your own models |
## Environment:
* Platform: Linux x86_64
* GPU: Nvidia GPUs
* CPUs
## Softwares
| Tool | Usage |
| :------------ |:---------------:|
| Guppy | ONT offical software to generate fastq through basecalling|
| minimap2      | align reads to reference.fa |
| samtools      | bam files processing |
| slow5tools      | converting (FAST5 <-> SLOW5) |
| f5c      | eventalign, assign current signals to bases |
## python modules
| Tool | Usage |
| :------------ |:---------------:|
| torch | an open source Python machine learning library|
# Installation
It may take several minutes to install  
```
git clone https://github.com/guodongxiaren/ImageCache/raw/master/Logo/foryou.gif
```
# Getting Started
## 1. Base calling and alignment
* Convert all Fast5 files into a single Blow5 file (slow5tools)
```
# convert a directory of fast5 files into BLOW5 files (default compression: zlib+svb-zd)
slow5tools f2s fast5_dir -d blow5_dir
# merge all BLOW5 files in a directory into a single BLOW5 file (default compression: zlib+svb-zd)
slow5tools merge blow5_dir -o file.blow5
```
* ONT guppy basecalling (guppy_basecaller)
```
# The input is 'fast5_dir', and the output is 'output_dir'. The input should be 'guppy_flowcell' and 'guppy_kit' depending on the sequencing platform.  
# Option: If you want to use a GPU, select '-x GPU'; the default is to use the CPU.
guppy_basecaller -i fast5_dir -s output_dir --flowcell guppy_flowcell --kit guppy_kit -x GPU -r
# Merge all fastq files,the output is 'combined_fastq'
cat output_dir/*.fastq > combined_fastq
```
* Alignment with reference.fa (minimap2)
```
# The input files contain reference file 'reference.fa' and basecalled fastq file 'combined_fastq', the output is sam file 'alignment_sam'
minimap2 -a reference.fa combined_fastq -ax map-ont > alignment_sam
```
* Convert and sort SAM to BAM (samtools)
```
# The input is sam file 'alignment_sam', the output is sorted bam file 'alignment_sorted_bam'
samtools view -S alignment_sam -b | samtools sort -o alignment_sorted_bam - ; samtools index alignment_sorted_bam
```
* Event alignment (f5c)
```
# The input files contain fastq file 'combined_fastq' and blow5 file 'file.blow5'
f5c index combined_fastq --slow5 file.blow5
# The files contain fastq file 'combined_fastq', reference file 'reference.fa', sorted bam file 'alignment_sorted_bam' and blow5 file 'file.blow5'. The output is csv file 'eventalign_output.csv'
f5c eventalign -r combined_fastq -g reference.fa -b alignment_sorted_bam --slow5 file.blow5 --rna --signal-index --print-read-name > eventalign_output.csv 
```
## 2. Ground truth alignment and data preprocessing
* Optional: If want to retrain the model using your own data, you need to align it with the ground truth.
```
# The input files contain your groundtruth file 'your_groundtruth_file.csv', finally output the processed file 'groundtruth_file'
python ground_truth_process.py -i 'your_groundtruth_file.csv'   
# The input files contain event alignment file 'eventalign_output.csv', template output folder 'template_output_folder', finally output folder 'output_file_folder' and groundtruth file 'groundtruth_file'
python align_label.py  -f eventalign_output.csv  -t template_output_folder -o output_file_folder -g groundtruth_file
# The input files contain raw current file 'file.blow5', template output folder 'template_output_folder', align_label output folder 'align_label_folder' and finally output folder 'output_file_folder'
python align_raw_current.py -b file.blow5 -t template_output_folder -a align_label_folder -o output_file_folder
```
* Required: Whether training or inference using your own data, feature extraction is necessary.
```
# The input files contain align_raw_current output folder 'output_file_folder' and template output folder 'template_output_folder', finally output file 'output_feature.feather'
python feature_extraction.py -a output_file_folder -t template_output_folder
```
## 3. Classification model training and inference
* Classification model training
```
# The input files contain feature file folder 'feature_folder', model save folder 'model_saved_folder'. Optional model training input parameters: -e (epochs, default 100), -b (batch size, default 512), -d (device, default cuda)
python Classification_model_training.py -f feature_folder -m model_saved_folder
```
* Classification model inference
```
# The input files contain feature file folder 'feature_folder', model save folder 'model_saved_folder' and inference result folder 'classification_inference_result_folder'. Optional model training input parameters: -b (batch size, default 512), -d (device, default cuda)
python Classification_model_inference.py -f feature_folder -m model_saved_folder -o classification_inference_result_folder
```
## 4. Regression model training and inference
* Regression model training
```
# The input files contain classification inference_result_folder 'classification_inference_result_folder', model save folder 'model_saved_folder' and the processed data template folder 'processed_data_template_floder'. Optional model training input parameters: -e (epochs, default 100), -d (device, default cuda)
python Regression_model_training.py -f classification_inference_result_folder -t processed_data_template_floder -m model_saved_folder
```
* Regression model inference
```
# The input files contain classification inference_result_folder 'classification_inference_result_folder', model save folder 'model_saved_folder' and inference result folder 'regression_inference_result_folder'. Optional model training input parameter:-d (device, default cuda)
python Regression_model_inference.py -f classification_inference_result_folder -m model_saved_folder -o regression_inference_result_folder
```
## 5. Clustering (single molecule label output)
```
# The input files contain regression inference_result_folder 'regression_inference_result_folder' and the single molecule result folder 'single_molecule_result_folder'. Optional model training input parameters: -e (epochs, default 100), -d (device, default cuda)
python Single_molecule_results.py -f regression_inference_result_folder -o single_molecule_result_folder
```
# Getting help
We appreciate your feedback and questions. You can report an error or suggestions related to Primepore as an issue on github.
