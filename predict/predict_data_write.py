from optparse import OptionParser
import os
import sys
import numpy as np
import pyBigWig
import pysam
import pandas as pd
from Roformer_data import ModelSeq
from dna_io import dna_1hot, dna_1hot_index
import tensorflow as tf
import h5py

"""
Roformer_data_write.py

Write TF Records for batches of model sequences.

"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_clip', dest='umap_clip',
      default=1, type='float',
      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
  parser.add_option('--umap_tfr', dest='umap_tfr',
      default=False, action='store_true',
      help='Save umap array into TFRecords [Default: %default]')
  parser.add_option('-x', dest='extend_bp',
      default=0, type='int',
      help='Extend sequences on each side [Default: %default]')
  parser.add_option('--idx', dest='idx',
      help='index of chr [Default: %default]')
  parser.add_option('--ref', dest='ref_genome',
      help='reference genome(*.fasta) [Default: %default]')
  (options, args) = parser.parse_args()
  
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide input arguments.')
  else:
    seqs_bed_file = args[0]
    atacseq_file = args[1]
    # roseq_plus_file = args[2]
    tfr_file = args[2]
    
  fasta_file = options.ref_genome

  chr_length_human = make_length_dict(options.idx)
  chrID_dict = make_chr_id(options.idx)

  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]), a[3]))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # write TFRecords
  ################################################################
  # user did not provide a reference genome file
  if fasta_file == 'None':
    # define options
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
    with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
      for si in range(num_seqs):
        # start and end of each 197k segment
        msi = options.start_i + si
        mseq = model_seqs[msi]
        mseq_start = mseq.start - options.extend_bp
        mseq_end = mseq.end + options.extend_bp
        target_length = mseq_end - mseq_start
        gap = (196608 - target_length) // 2###40960
        mseq_start = mseq_start - gap
        mseq_end = mseq_end + gap

        # use a 197 kbp 'AAAA' represents sequence
        seq_dna = 'A'*196608
        seq_1hot = dna_1hot(seq_dna, n_uniform=False, n_sample=False)

        # read RO-seq signal
        atac_seq = np.asarray(get_atac_seq(atacseq_file, mseq.chr, mseq_start, mseq_end, chr_length_human))

        # absolute value
        atac_seq = abs(atac_seq)

        # remove abnormal values
        atac_seq[np.where(np.isnan(atac_seq))] = 1e-3
        atac_seq[np.where(np.isinf(atac_seq))] = 1e-3

        # record start and end of each segment
        start_end = [chrID_dict[mseq.chr], mseq_start, mseq_end]
        start_end = np.asarray(start_end).astype('int32')

        # hash to bytes
        features_dict = {
          'sequence': feature_bytes(seq_1hot),
          'atac-seq': feature_bytes(atac_seq),
          'start-end': feature_bytes(start_end)
          }

        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        writer.write(example.SerializeToString())
  # user provided a reference genome file   
  else:
    # open FASTA
    fasta_open = pysam.Fastafile(fasta_file)

    # define options
    tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

    with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
      for si in range(num_seqs):
        # start and end of each 197k segment
        msi = options.start_i + si
        mseq = model_seqs[msi]
        mseq_start = mseq.start - options.extend_bp
        mseq_end = mseq.end + options.extend_bp
        target_length = mseq_end - mseq_start
        gap = (196608 - target_length) // 2
        mseq_start = mseq_start - gap
        mseq_end = mseq_end + gap

        # read DNA sequence, and convert sequence to one-hot encoding
        seq_dna = fetch_dna(fasta_open, mseq.chr, mseq_start, mseq_end)
        seq_1hot = dna_1hot(seq_dna, n_uniform=False, n_sample=False)

        # read RO-seq signal
        atac_seq = np.asarray(get_atac_seq(atacseq_file, mseq.chr, mseq_start, mseq_end, chr_length_human))

        # absolute value
        atac_seq = abs(atac_seq)

        # remove abnormal values
        atac_seq[np.where(np.isnan(atac_seq))] = 1e-3
        atac_seq[np.where(np.isinf(atac_seq))] = 1e-3

        # record start and end of each segment
        start_end = [chrID_dict[mseq.chr], mseq_start, mseq_end]
        start_end = np.asarray(start_end).astype('int32')

        # hash to bytes
        features_dict = {
          'sequence': feature_bytes(seq_1hot),
          'atac-seq': feature_bytes(atac_seq),
          'start-end': feature_bytes(start_end)
          }

        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        writer.write(example.SerializeToString())

      fasta_open.close()


def get_atac_seq(seq_file, chr, start, end, chr_length):
  """
  Read atac_seq signal

  Args:
        seq_file: atac.bw
        chr: chromosome, eg. chr1
        start: start of this segment
        end: end of this segment
        chr_length: a dict recorded length of each chromosome of the reference genome

  Output:
        atac-seq: atac-seq signal
  """
  atac_seq = []

  genome_cov_file = seq_file
  # genome_cov_file_plus = roseq_plus_file

  # build CovFace object
  try:
    genome_cov_open = CovFace(genome_cov_file)
    # genome_cov_open_plus = CovFace(genome_cov_file_plus)
  except:
    print('an error when read ',genome_cov_file)
    exit()
  
  # judge if the line is crossed
  p_start = start if start > 0 else 0
  p_end = end if end < chr_length[chr] else chr_length[chr]

  # read file
  try:
    seq_cov_nt = genome_cov_open.read(chr, p_start, p_end)
    # seq_cov_nt_plus = genome_cov_open_plus.read(chr, p_start, p_end)
  except:
    print(chr, start, end)
    print(chr, p_start, p_end)
    exit()

  # remove abnormal values
  baseline_cov = np.percentile(seq_cov_nt, 100*0.5)
  baseline_cov = np.nan_to_num(baseline_cov)
  nan_mask = np.isnan(seq_cov_nt)
  seq_cov_nt[nan_mask] = baseline_cov

  # concatenate the out-of-bounds part and assign a value of 0 to the out-of-bounds part
  seq_cov_nt = np.hstack((np.zeros(abs(start-p_start)), seq_cov_nt))
  seq_cov_nt = np.hstack((seq_cov_nt, np.zeros(abs(end-p_end)))).astype('float16')
  atac_seq.append(seq_cov_nt)


  return atac_seq


def fetch_dna(fasta_open, chrm, start, end):
  """Fetch DNA when start/end may reach beyond chromosomes."""

  # initialize sequence
  seq_len = end - start
  seq_dna = ''

  # add N's for left over reach
  if start < 0:
    seq_dna = 'N'*(-start)
    start = 0

  # get dna
  seq_dna += fasta_open.fetch(chrm, start, end)

  # add N's for right over reach
  if len(seq_dna) < seq_len:
    seq_dna += 'N'*(seq_len-len(seq_dna))

  return seq_dna


def feature_bytes(values):
  """Convert numpy arrays to bytes features."""
  values = values.flatten().tobytes()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def feature_floats(values):
  """Convert numpy arrays to floats features.
     Requires more space than bytes."""
  values = values.flatten().tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def normali(Z):
  norm = np.linalg.norm(Z, axis=0)
  out = Z/norm
  return out


def make_length_dict(path):
  length_dict = {}
  for line in open(path):
    a = line.split()
    length_dict[a[0]] = int(a[2])
  return length_dict


def make_chr_id(path):
  id_dict = {}
  for line in open(path):
    a = line.split()
    id_dict[a[0]] = a[4]
  return id_dict

class CovFace:
  def __init__(self, cov_file):
    self.cov_file = cov_file
    self.bigwig = False
    self.bed = False

    cov_ext = os.path.splitext(self.cov_file)[1].lower()
    if cov_ext == '.gz':
      cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

    if cov_ext in ['.bed', '.narrowpeak']:
      self.bed = True
      self.preprocess_bed()

    elif cov_ext in ['.bw','.bigwig']:
      self.cov_open = pyBigWig.open(self.cov_file, 'r')
      self.bigwig = True

    elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
      self.cov_open = h5py.File(self.cov_file, 'r')

    else:
      print('Cannot identify coverage file extension "%s".' % cov_ext,
            file=sys.stderr)
      exit(1)

  def preprocess_bed(self):
    # read BED
    bed_df = pd.read_csv(self.cov_file, sep='\t',
      usecols=range(3), names=['chr','start','end'])

    # for each chromosome
    self.cov_open = {}
    for chrm in bed_df.chr.unique():
      bed_chr_df = bed_df[bed_df.chr==chrm]

      # find max pos
      pos_max = bed_chr_df.end.max()

      # initialize array
      self.cov_open[chrm] = np.zeros(pos_max, dtype='bool')

      # set peaks
      for peak in bed_chr_df.itertuples():
        self.cov_open[peak.chr][peak.start:peak.end] = 1


  def read(self, chrm, start, end):
    if self.bigwig:
      cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')
    else:
      if chrm in self.cov_open:
        cov = self.cov_open[chrm][start:end]
        pad_zeros = end-start-len(cov)
        if pad_zeros > 0:
          cov_pad = np.zeros(pad_zeros, dtype='bool')
          cov = np.concatenate([cov, cov_pad])
      else:
        print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
          (self.cov_file, chrm, start, end), file=sys.stderr)
        cov = np.zeros(end-start, dtype='float16')

    return cov

  def close(self):
    if not self.bed:
      self.cov_open.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  # pass
