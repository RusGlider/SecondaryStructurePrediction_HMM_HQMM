from Bio import SeqIO
from os import listdir
import numpy as np
from hmmlearn import hmm
from utils import *
import pickle

from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

import itertools

pdb_parser = PDBParser()
cif_parser = MMCIFParser()

aa_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'Y', 'Z', 'X', '*']

aa_list2 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
aa_pairs_list = [ ''.join(i) for i in itertools.product(aa_list,repeat=2)]


def load_model(filename):
    with open(filename, "rb") as file:
        model =  pickle.load(file)
    return model

def save_model(model,name):
    with open(name+'.pkl', "wb") as file: pickle.dump(model, file)


# listing 1
def read_sequences(fasta_dir, dssp_dir, n_sequences=1):
    #read sequences and its dssp markings and return in pairs [seq,seq]
    fasta_files = listdir(fasta_dir)[:n_sequences]
    dssp_files = listdir(dssp_dir)[:n_sequences]
    fasta_dssp_pairs = []
    for i in range(n_sequences):
        sequence = SeqIO.read(fasta_dir+'/'+fasta_files[i],'fasta')
        dssp = SeqIO.read(dssp_dir+'/'+dssp_files[i],'fasta')
        pair = [sequence, dssp]
        fasta_dssp_pairs.append(pair)
    return fasta_dssp_pairs

# listing 2

def get_secondary_structures(sequence, dssp):
    # get all marked secondary structures from sequence
    helices = get_sub_by_char(sequence, dssp, 'H')
    strands = get_sub_by_char(sequence, dssp, 'E')
    turns = get_sub_by_char(sequence, dssp, 'T')
    other = get_sub_by_char(sequence, dssp, '-')
    return helices, strands, turns, other

def create_secondary_structure_dataset(pairs):
    #create dataset of helices, strands and turns from pairs of fasta and dssp sequences
    helix_dataset = []
    strand_dataset = []
    turn_dataset = []
    other_dataset = []
    for pair in pairs:
        helix, strand, turn, other = get_secondary_structures(pair[0], pair[1])
        helix_dataset += helix
        strand_dataset += strand
        turn_dataset += turn
        other_dataset += other
    return helix_dataset, strand_dataset, turn_dataset, other_dataset # ['abc','def','ghi']


def convert_aa_to_num(seq):
    #convert amino-acids names to integers
    aa_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'Y', 'Z', 'X', '*']
    out = [aa_list.index(c) for c in seq]
    return out

def convert_num_to_aa(seq):
    #convert numbers to respective amino-acids
    aa_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'Y', 'Z', 'X', '*']
    out = [aa_list[i] for i in seq]
    return out

def prepare_dataset_for_hmm(dataset):
    #converts a list of sequences to a np array ready to work in HMM
    out = [convert_aa_to_num(seq) for seq in dataset] #converting chars to integers
    out = [[[c] for c in list(seq)] for seq in out] #transorming shape of dataset for hmmlearn
    out_lengths = np.array([len(seq) for seq in out]) #getting lengths of sequences for hmmlearn
    out = np.concatenate(out)
    out.reshape(-1, 1)
    return out, out_lengths


# ==================== bigram dataset

def convert_seq_to_aa_pairs(seq):
    #converts sequence to sequence of pairs. ABCD -> AB BC CD
    # discards sequences of length 1

    out = [seq[i]+seq[i+1] for i in range(len(seq)-1)]
    return out

def convert_aa_pairs_to_nums(seq):
    #converts sequence of pairs to numbers. AB BC CD -> 1 2 3
    out = [aa_pairs_list.index(c) for c in seq]
    return out

def convert_nums_to_aa_pairs(seq):
    out = [aa_pairs_list[i] for i in seq]
    return out

def prepare_bidataset(dataset):
    #converts a list of sequences to a np array of bigrams ready to work in HMM

    out = [convert_seq_to_aa_pairs(seq) for seq in dataset if len(seq) > 1]
    out = [convert_aa_pairs_to_nums(seq) for seq in out]
    out = [[[c] for c in list(seq)] for seq in out]
    out_lengths = np.array([len(seq) for seq in out])
    out = np.concatenate(out)
    out.reshape(-1, 1)
    return out, out_lengths

# ====================


def fit_model(n_fits,train,train_lengths, validate, validate_lengths):
    np.random.seed(0)
    best_score = best_model = None
    for i in range(n_fits):
        model = hmm.CategoricalHMM(n_components=2, random_state=i).fit(train,train_lengths)
        score = model.score(validate,validate_lengths)
        print(f'Model #{i}\tScore: {score}')
        if best_score is None or score > best_score:
            best_model = model
            best_score = score
    return best_model



if __name__ == "__main__":
    print(len(aa_pairs_list))
    fasta_dir = 'data/fasta'
    dssp_dir = 'data/dssp'
    fasta_dssp_pairs = read_sequences(fasta_dir, dssp_dir, 500)
    helix_dataset, strand_dataset, turn_dataset = create_secondary_structure_dataset(fasta_dssp_pairs)

    dataset = np.array(helix_dataset)
    X_train = helix_dataset[:dataset.shape[0] // 2]
    X_validate = helix_dataset[dataset.shape[0] // 2:]

    X_train, X_train_lengths = prepare_dataset_for_hmm(X_train)
    X_validate, X_validate_lengths = prepare_dataset_for_hmm(X_validate)

    Y_validate, Y_validate_lengths = prepare_dataset_for_hmm(np.array(strand_dataset))

    helix_model = fit_model(20,X_train,X_train_lengths,X_validate,X_validate_lengths)

    #model = hmm.CategoricalHMM(n_components=2).fit(X_train, X_train_lengths)

    helix_pred = helix_model.predict(X_validate, X_validate_lengths)
    falsehelix_pred = helix_model.predict(Y_validate, Y_validate_lengths)

    acc1 = np.count_nonzero(helix_pred) / len(helix_pred)
    acc2 = ( len(falsehelix_pred)-np.count_nonzero(falsehelix_pred) ) / len(falsehelix_pred)
    print('helix prediction: ', helix_pred)
    print('accuracy: ', acc1)
    print('false-helix prediction (strands): ', falsehelix_pred)
    print('accuracy: ', acc2)
    print()













