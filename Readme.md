**Note: ** This repository is part of the assignment given in [Information Communication Theory (情報伝達学)](http://www.cl.ecei.tohoku.ac.jp/index.php?InformationCommunicationTheory) lecture.

# About

This is the implementation of Named Entitty Recognition (NER) model based on Recurrent Neural Network (RNN). The model is heavily inspired by following papers:

* Chiu, Jason PC, and Eric Nichols. "Named entity recognition with bidirectional LSTM-CNNs." Transactions of the Association for Computational Linguistics 4 (2016): 357-370.
* James Hammerton. "Named Entity Recognition with Long Short-Term Memory." CONLL '03 Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003 - Volume 4
Pages 172-175
* Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami and Chris Dyer. "Neural Architectures for Named Entity Recognition." Proceedings of NAACL-HLT 2016, pages 260–270

Note that this repo is not re-implementation of these models.

# Model Details
Following models are implemented with Chainer.

## Models with Cross Entropy as Loss Function
* LSTM (Model.py/NERTagger)
* Bi-directional LSTM (Model.py/BiNERTagger)
* Bi-directional LSTM with Character-based encoding (Model.py/BiCharNERTagger)

## Models with CRF Layer as Loss Function
* LSTM (CRFModel.py/CRFNERTagger)
* Bi-directional LSTM (CRFModel.py/CRFBiNERTagger)
* Bi-directional LSTM with Character-based encoding (CRFModel.py/CRFBiCharNERTagger)

# Requirements
## Software

* Python 3.*
* Chainer 1.19 (or higher)

## Resources

* Pretrained Word Vector (e.g. [GloVe](http://nlp.stanford.edu/projects/glove/))
    * The script will still work without these vectors, but the performance will significantly deteriorate.
* [CoNLL 2003 Dataset](http://www.cnts.ua.ac.be/conll2003/ner/)

# Usage
## Preprocessing
Place CoNLL datasets `train dev test` in `data/` and run `preprocess.sh`. This converts raw datasets into model-readable json format.

Then run `generate_vocab.py` and `generate_char_vocab.py` to generate vocabulary files.

## Training


## Testing

# Result
TBA
