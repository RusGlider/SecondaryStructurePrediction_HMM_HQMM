from PDB_import import *
from utils import *
from hmmlearn import hmm
import random
import time
import matplotlib.pyplot as plt
import QExperimentLauncher



def custom_score(model,validate,validate_lengths):
    predictions = model.predict(validate,validate_lengths)
    score = np.count_nonzero(predictions) / len(predictions)
    return score

def custom_score_length(model,validate,validate_lengths):
    # Оценка модели по нескольким параметрам.
    predictions = model.predict(validate, validate_lengths)
    mean_validate_len = sum(validate_lengths) / len(validate_lengths)
    mean_validate_count = len(validate_lengths)

    test_lengths = []
    clen = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            clen+=1
        elif predictions[i] == 0:
            if clen > 0:
                test_lengths.append(clen)
            clen = 0
    mean_test_len = sum(test_lengths) / len(test_lengths)
    mean_test_count = len(test_lengths)

    len_ratio = mean_test_len / mean_validate_len
    count_ratio = mean_validate_count / mean_test_count
    pred_ratio = (np.count_nonzero(predictions) / len(predictions))

    if (len_ratio > 1):
        len_ratio = max(1 - (len_ratio-1),0)
    if (count_ratio > 1):
        count_ratio = max(1 - (count_ratio - 1), 0)

    print(f'pred ratio: {pred_ratio}, len_ratio: {len_ratio}, count_ratio: {count_ratio}')
    score = (pred_ratio + len_ratio + count_ratio) / 3
    if (score > 1 or len_ratio > 1):
        print('score > 1')
    return score

def custom_false_score(model,validate,validate_lengths,false_validate,false_validate_lengths):
    score = custom_score(model,validate,validate_lengths)
    neg_score = custom_score(model,false_validate,false_validate_lengths)
    return score, 1-neg_score

def custom_multiscore(model,validates,validates_lengths,valid_preds):
    scores = []
    for i in range(len(validates)):
        predictions = model.predict(validates[i],validates_lengths[i])
        score = list.count(predictions,valid_preds[i])
        scores.append(score)
    return scores




def score_satisfies(score,last,max):
    return (last is None or score > last) and score < max

def fit_model(n_fits,train,train_lengths, validate, validate_lengths, false_validate=None, false_validate_lengths=None,min_score = 1.0, max_score = 1.0):
    #np.random.seed(0)
    best_score = best_model = best_neg_score = None
    for i in range(n_fits):
        model = hmm.CategoricalHMM(n_components=2, random_state=i,implementation='scaling').fit(train,train_lengths)
        #model = hmm.GaussianHMM(n_components=2,random_state=i).fit(train,train_lengths)
        #score = model.score(validate,validate_lengths)
        if false_validate is None or false_validate_lengths is None:
            score = custom_score_length(model,validate,validate_lengths)
            print(f'Model #{i}\tScore: {score}')
            if score_satisfies(score,best_score,max_score):
                best_model = model
                best_score = score
            if best_score > min_score:
                print(f'Score threshold [{min_score};{max_score}] satisfied')
                break
        else:
            score, neg_score = custom_false_score(model,validate,validate_lengths,false_validate,false_validate_lengths)
            print(f'Model #{i}\tScore: {score}, Neg score: {neg_score}')
            if score_satisfies(score,best_score,max_score) and score_satisfies(neg_score,best_neg_score,max_score):
                best_model = model
                best_score = score
                best_neg_score = neg_score
            if best_score > min_score or best_neg_score > min_score:
                print(f'Score threshold [{min_score};{max_score}] satisfied')
                break
    print('Best score: ',best_score)
    return best_model



def train_model(dataset, train_ratio=0.5, n_fits=10, min_score=0.95, max_score = 1.0):
    # trains the model on given dataset
    train, validate = split_list_by_ratio(dataset,train_ratio)
    train_input, train_lengths = prepare_dataset_for_hmm(train)
    validate_input, validate_lengths = prepare_dataset_for_hmm(validate)
    model = fit_model(n_fits,train_input,train_lengths,validate_input,validate_lengths,min_score=min_score, max_score = max_score)
    return model

def train_bi_model(dataset, train_ratio=0.5, n_fits=10, min_score=1.0, max_score = 1.0,false_dataset=None):
    train, validate = split_list_by_ratio(dataset,train_ratio)
    train_input, train_lengths = prepare_bidataset(train)
    validate_input, validate_lengths = prepare_bidataset(validate)
    false_validate = false_validate_lengths = None
    if false_dataset is not None:
        false_validate, false_validate_lengths = prepare_bidataset(false_dataset)
    model = fit_model(n_fits, train_input, train_lengths, validate_input, validate_lengths,min_score=min_score, max_score = max_score,false_validate=false_validate, false_validate_lengths=false_validate_lengths)
    return model


def bigram_test():
    fasta_dir = 'data/fasta'
    dssp_dir = 'data/dssp'
    fasta_dssp_pairs = read_sequences(fasta_dir, dssp_dir, 1000)
    helix_dataset, strand_dataset, turn_dataset = create_secondary_structure_dataset(fasta_dssp_pairs)


def combine_models_predictions(helix_pred, strand_pred, other_pred):
    # Совмещение оценок разметки в одну совокупную разметку
    combined = ''
    clen = 0
    helix_count = 0
    strand_count = 0
    for i in range(len(other_pred)):

        if other_pred[i] == 'X':

            clen += 1

            if helix_pred[i] == 'H':
                helix_count += 1
            if strand_pred[i] == 'E':
                strand_count += 1
        else:

            if clen > 0:
                if helix_count >= strand_count:
                    combined += 'H' * clen
                else:
                    combined += 'E' * clen
                clen = 0
                helix_count = 0
                strand_count = 0

            combined += '-'
    return combined

def test_and_load_models(helixfile,strandfile,otherfile, testfile):
    # Загрузка моделей и их тестирование на отдельной последовательности
    helix_model = load_model(helixfile)
    strand_model = load_model(strandfile)
    other_model = load_model(otherfile)

    sequence = SeqIO.read('data/fasta/' + testfile + '.fasta', 'fasta')
    dssp = SeqIO.read('data/dssp/' + testfile + '.dssp', 'fasta')
    fa = [str(sequence.seq)]
    dp = [str(dssp.seq)]

    t, tlen = prepare_bidataset(fa)
    hpred = helix_model.predict(t, tlen)
    spred = strand_model.predict(t, tlen)
    opred = other_model.predict(t, tlen)
    print(hpred)
    helix_assumption = ''.join([('H' if i == 1 else '-') for i in hpred])
    strand_assumption = ''.join([('E' if i == 1 else '-') for i in spred])
    other_assumption = ''.join([('-' if i == 1 else 'X') for i in opred])
    print('fasta:', fa)
    print('dssp:', dp)
    print('helix: ', helix_assumption)
    print('strand:', strand_assumption)
    print('other:', other_assumption)
    print('combined:',combine_models_predictions(helix_assumption,strand_assumption,other_assumption))

def training_experiment():
    fasta_dir = 'data/fasta'
    dssp_dir = 'data/dssp'
    fasta_dssp_pairs = read_sequences(fasta_dir, dssp_dir, 1000)
    helix_dataset, strand_dataset, turn_dataset, other_dataset = create_secondary_structure_dataset(fasta_dssp_pairs)

    print('learning helix model...')
    helix_model = train_model(helix_dataset, n_fits=50, min_score=0.80, max_score=0.9)
    print('learning strand model...')
    strand_model = train_model(strand_dataset, n_fits=50, min_score=0.80, max_score=0.9)
    print('learning other model...')
    other_model = train_model(other_dataset, n_fits=50, min_score=0.85, max_score=0.9)



    fa = ['MNLTELKNTPVSELITLGENMGLENLARMRKQDIIFAILKQHAKSGE']
    dp = ['-EHHHHH---HHHHHHHHH-----------HHHHHHHHHHHHHH---']
    t, tlen = prepare_dataset_for_hmm(fa)
    helix_assumption = ''.join([('H' if i == 1 else '-') for i in helix_model.predict(t,tlen)])
    strand_assumption = ''.join([('E' if i == 1 else '-') for i in strand_model.predict(t,tlen)])
    other_assumption = ''.join([('-' if i == 1 else 'X') for i in other_model.predict(t,tlen)])
    print('dssp:',dp)
    print('helix: ',helix_assumption)
    print('strand:', strand_assumption)
    print('other:',other_assumption)

    print("save models? Y/N")
    ans = input()
    if ans.lower() == 'y':
        pref = input('prefix:')
        save_model(helix_model, 'helix_model'+pref)
        save_model(strand_model, 'strand_model'+pref)
        save_model(other_model, 'other_model'+pref)
    else:
        print('models are not saved')

    #separate_experiment()

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()




def test_models(helix_model,strand_model,other_model):
    fasta_dir = 'data/fasta'
    dssp_dir = 'data/dssp'
    fasta_files = listdir(fasta_dir)
    dssp_files = listdir(dssp_dir)

    accuracies = []
    helix_accs = [] ; helix_times = []
    strand_accs = [] ; strand_times = []
    other_accs = [] ; other_times = []
    for i in range(len(fasta_files)): #
        try:
            sequence = [str(SeqIO.read(fasta_dir + '/' + fasta_files[i], 'fasta').seq)]
            dssp = [str(SeqIO.read(dssp_dir + '/' + dssp_files[i], 'fasta').seq)]
            #test, test_len = prepare_bidataset(sequence)
            test, test_len = prepare_dataset_for_hmm(sequence)

            t1 = time.time()
            helix_assumption = ''.join([('H' if i == 1 else '-') for i in helix_model.predict(test, test_len)])
            t2 = time.time()
            helix_times.append([(t2-t1) * 10**3,test_len[0]])

            t1 = time.time()
            strand_assumption = ''.join([('E' if i == 1 else '-') for i in strand_model.predict(test, test_len)])
            t2 = time.time()
            strand_times.append([(t2-t1) * 10**3,test_len[0]])

            t1 = time.time()
            other_assumption = ''.join([('-' if i == 1 else 'X') for i in other_model.predict(test, test_len)])
            t2 = time.time()
            other_times.append([(t2-t1) * 10**3,test_len[0]])
            assumption = combine_models_predictions(helix_assumption, strand_assumption, other_assumption)
            accuracy = similar(assumption, dssp[0])
            helix_accuracy = similar(''.join([('H' if s == 'H' else '-') for s in assumption]),''.join([('H' if s == 'H' else '-') for s in dssp[0]]))
            strand_accuracy = similar([('E' if s == 'E' else '-') for s in assumption],[('E' if s == 'E' else '-') for s in dssp[0]])
            other_accuracy = similar(other_assumption,[('X' if s in ['H','E'] else '-') for s in dssp[0]])
            accuracies.append(accuracy)
            helix_accs.append(helix_accuracy)
            strand_accs.append(strand_accuracy)
            other_accs.append(other_accuracy)
            print(f'============================== Sequence {i + 1}/{len(fasta_files)}')
            print('dssp     :', dssp[0])
            print('predicted:', assumption)
            print()
            print('overall accuracy :', accuracy)

            print(f'helix| accuracy: {helix_accuracy}; time: {helix_times[-1][0]} ms, len: {test_len[0]}')
            print(f'strand| accuracy: {strand_accuracy}; time: {strand_times[-1][0]} ms, len: {test_len[0]}')
            print(f'other| accuracy: {other_accuracy}; time: {other_times[-1][0]} ms, len: {test_len[0]}')
            print(f'')
        except:
            print('unable to predict sequence')
            continue

    print('==================================================')
    print('average accuracy:',sum(accuracies) / len(accuracies) )
    print('min accuracy:', min(accuracies))
    print('max accuracy:', max(accuracies))
    print()
    print(f'helix | avg accuracy: {sum(helix_accs) / len(helix_accs)}; avg time: {sum(t for t, acc in helix_times) / len(helix_times)}')
    print(f'strand | avg accuracy: {sum(strand_accs) / len(strand_accs)}; avg time: {sum(t for t, acc in strand_times) / len(strand_times)}')
    print(f'other | avg accuracy: {sum(other_accs) / len(other_accs)}; avg time: {sum(t for t, acc in other_times) / len(other_times)}')

    return accuracies, helix_times, strand_times, other_times





def test_quantum_model(helix_model,strand_model,other_model):
    fasta_dir = 'data/fasta'
    dssp_dir = 'data/dssp'
    fasta_files = listdir(fasta_dir)[:100]
    dssp_files = listdir(dssp_dir)[:100]

    accuracies = []
    helix_accs = []
    helix_times = []
    strand_accs = []
    strand_times = []
    other_accs = []
    other_times = []
    for i in range(len(fasta_files)): #len(fasta_files)
        try:
            sequence = str(SeqIO.read(fasta_dir + '/' + fasta_files[i], 'fasta').seq)
            dssp = str(SeqIO.read(dssp_dir + '/' + dssp_files[i], 'fasta').seq)

            #seq_pairs = convert_seq_to_aa_pairs(sequence)
            #ready_seq = convert_aa_pairs_to_nums(seq_pairs)
            ready_seq = convert_aa_to_num(sequence)

            t1 = time.time()
            helix_pred = QExperimentLauncher.Run_HQMM(helix_model,ready_seq)
            t2 = time.time()
            helix_times.append([(t2 - t1) * 10 ** 3, len(sequence)])

            t1 = time.time()
            strand_pred = QExperimentLauncher.Run_HQMM(strand_model,ready_seq)
            t2 = time.time()
            strand_times.append([(t2 - t1) * 10 ** 3, len(sequence)])

            t1 = time.time()
            other_pred = QExperimentLauncher.Run_HQMM(other_model,ready_seq)
            t2 = time.time()
            other_times.append([(t2 - t1) * 10 ** 3, len(sequence)])

            helix_assumption = ''.join([('H' if i > 0 else '-') for i in helix_pred])
            strand_assumption = ''.join([('E' if i > 0 else '-') for i in strand_pred])
            other_assumption = ''.join([('-' if i > 0 else 'X') for i in other_pred])

            assumption = combine_models_predictions(helix_assumption, strand_assumption, other_assumption)
            accuracy = similar(assumption, dssp)
            helix_accuracy = similar(''.join([('H' if s == 'H' else '-') for s in assumption]),
                                     ''.join([('H' if s == 'H' else '-') for s in dssp]))
            strand_accuracy = similar([('E' if s == 'E' else '-') for s in assumption],
                                      [('E' if s == 'E' else '-') for s in dssp])
            other_accuracy = similar(other_assumption, [('X' if s in ['H', 'E'] else '-') for s in dssp])

            accuracies.append(accuracy)
            helix_accs.append(helix_accuracy)
            strand_accs.append(strand_accuracy)
            other_accs.append(other_accuracy)
            print(f'============================== Sequence {i + 1}/{len(fasta_files)}')
            print('dssp     :', dssp[0])
            print('predicted:', assumption)
            print()
            print('overall accuracy :', accuracy)

            print(f'helix| accuracy: {helix_accuracy}; time: {helix_times[-1][0]} ms, len: {len(sequence)}')
            print(f'strand| accuracy: {strand_accuracy}; time: {strand_times[-1][0]} ms, len: {len(sequence)}')
            print(f'other| accuracy: {other_accuracy}; time: {other_times[-1][0]} ms, len: {len(sequence)}')
        except:
            print('unable to predict sequence')
            continue

    print('==================================================')
    print('average accuracy:', sum(accuracies) / len(accuracies))
    print('min accuracy:', min(accuracies))
    print('max accuracy:', max(accuracies))
    print()
    print(f'helix | avg accuracy: {sum(helix_accs) / len(helix_accs)}; avg time: {sum(t for t, acc in helix_times) / len(helix_times)}')
    print(f'strand | avg accuracy: {sum(strand_accs) / len(strand_accs)}; avg time: {sum(t for t, acc in strand_times) / len(strand_times)}')
    print(f'other | avg accuracy: {sum(other_accs) / len(other_accs)}; avg time: {sum(t for t, acc in other_times) / len(other_times)}')

    return accuracies, helix_times, strand_times, other_times



if __name__ == "__main__":
    helix_model = load_model('helix_model_unigram.pkl')
    strand_model = load_model('strand_model_unigram.pkl')
    other_model = load_model('other_model_unigram.pkl')
    #accs, h_times, s_times, o_times = test_models(helix_model, strand_model, other_model)
    accs, h_times, s_times, o_times = test_quantum_model(helix_model, strand_model, other_model)


    h_time = [t for t, l in h_times]
    s_time = [t for t, l in s_times]
    o_time = [t for t, l in o_times]

    h_lengths = [l for t, l in h_times]
    s_lengths = [l for t, l in s_times]
    o_lengths = [l for t, l in o_times]


    #plt.plot('Длина последовательности','Время работы',data=None)
    plt.xlabel('Длина последовательности')
    plt.ylabel('Время работы, мс')

    plt.scatter(h_lengths, h_time)
    plt.show()

    plt.scatter(s_lengths, s_time)
    plt.show()

    plt.scatter(o_lengths, o_time)
    plt.show()

    #print(similar('----------HHHHHHH--EE-------HHHHHHHHHH--E---E-','----------EEEEEEEEEE---HHHHHHHHHHHH-----EEEEE-'))

    #test_and_load_models('helix_modelnew.pkl','strand_modelnew.pkl','other_modelnew.pkl','e1n13.1A')
    #training_experiment()
    #print(convert_aa_to_num('NLIRI'))

    """
    transmat = np.array([
        [0.9,0.1],
        [0.1,0.9]
    ])
    powered = transmat
    for i in range(5):
        powered = np.linalg.matrix_power(transmat,i)
        print(powered)
    """







