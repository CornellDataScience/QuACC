# Progress Report

02/28 (Yuji):
* Reading: [SQuAD](https://arxiv.org/pdf/1606.05250.pdf)
* Created `process.py`
  * Organize raw JSON data into tabular format. `data/raw_questions.csv`
  contains 87599 question/answer pairs, where each row records
  the article title, paragraph number, question, answer, and
  answer pointer. None of the text has been vectorized.
  * Sample usage: `python process.py data/train-v1.1.json`

02/28 (Kenta):
* Downloaded pre-trained glove to the team server
* Created `glove.py`
  * Load pre-trained model.
  * Return vectorized word if it exists in the vocabulary, `None` otherwise.
  * Sample usage:
  ```python
  from word_embed import Glove
  model = Glove()
  model.vectorize('hello')
  ```

03/01 (Yuji):
* Added `tokenize()` in `process.py`
  * Tokenize all answers using spaCy. This splits all answers into
  individual words (with intelligent constraints such as U.S.A. should
  not be split into three words), and also does POS tagging, lemmatization,
  dependency tree construction, etc.
  See [documentation](https://spacy.io/usage/spacy-101) for details.
  * Notes:
    * Tokenizing process should be parallelized (but isn't yet)
    * Resulting file is somewhat large (approx. 500MB)
* Some analysis of what most answers are like
  [here](https://github.com/CornellDataScience/NLP_Research-SP18/tree/master/qa_transfer/analysis/answer_distribution.ipynb).

03/02 (Kenta):
* Downloaded pre-trained Word2Vec to the team server
* Renamed `glove.py` to `pretrained_word_models.py`
  * Load pre-trained model.
  * Return vectorized word if it exists in the vocabulary, `None` otherwise.
  * Sample usage:
  ```python
  from word_embed import Word2Vec
  model = Word2Vec()
  model.vectorize('hello')
  ```

03/05 (Kenta, Yuji)
* Train on dependency tree generated by SpaCy?
* R-Net implementation embeds all OOV tokens as 0-vectors. Can we create
better initializations using SpaCy named entity recognition? Recursively
feed OOV tokens into R-Net?  
(e.g. If unknown token is "X", feed first paragraph of article "X" on
Wikipedia into R-Net and ask "What is X"?) 

03/06 (Kenta, Yuji)
* Created `char_embed.py`
  * Character embeddings used by [this](https://github.com/minsangkim142/R-net)
  implementation of R-Net uses character-level embeddings that
  are just weighted averages of every word vector in which
  the character appears.
  * Created character embeddings from pre-trained GloVe.
  
03/08 (All)
* Created `layers.py`, `model.py`
  * Potential exploration topic: why does R-Net only feed the last
  hidden layer of the bi-directional RNN for character embeddings?
  * Implemented `encoding` (does embedding lookup for words, chars)
  and `bidirectional_rnn`.
* Created `hyperparams.py`
  * Store all hyperparameters as a class.
* Created `loader.py`
  * Encode all context paragraphs and save
  * From raw_questions.csv, read article name & paragraph number,
  look up encoded paragraph
  * Encode question
  * Find answer start index & answer end index
* Schedule for next week:
  * Read about pointer net (alternatives ?)
  * Re-read R-Net, BiDAF, Mnemonic Reader papers
  * Exploratory analysis of SQuAD
  * Presentation
  
03/09 (Kenta)
* Connected the character embedding input to bidirectional_rnn layers.
  * Takenized characters will be first encoded to GloVe embedding, and 
  then the model takes the last hidden layer of bi-directional GRU unit.
  
03/10 (Yuji)
* Moved loading GloVe model to `util.py`
  * If `load_pretrained=False`, should embedding matrix be
  initialized to identity matrix? (i.e. one-hot encoding)

03/15 (Yuji)
* Added R-Net architecture diagram to README
  * Some hyperparameters are ambiguous: number of layers in
  self-matching network and gated attention-based GRU.
  * Formula (11) in the paper contains a mystery parameter
  V_r^Q.
* Edited `layers.py`
  * Return both final states and outputs in `bidirectional_rnn()`
  function: selecting value to pass through network should 
  be implemented in `model.py`
  * Preliminary implementation of attention layer; likely 
  needs revisions.
  
03/22 (Kenta)
* Generated complete dictionaries for all context paragraphs
  * Edited `process.py`
* Connected initial layers for both questions and passages
  * Edited `model.py`
 
03/23 (Kenta, Yuji)
* Started implementing attention matching
  * Discovered an error in the original paper and previous implementations of RNet
  * Came up with a potential solution:
    * Interpret "final state" of bidirectional RNN in character
    embedding as final hidden state at last character of word.

03/24 (Kenta)
* Created `cells.py` for the custom RNN cells
  * TODO 1: Insert the attention layer (to be implemented by Yuji) between previous states and current inputs
  * TODO 2: Implement GRU operation and wrap it around the attention layer
  * TODO 3: Return hidden layers inside of the output tuple 
* I was there when Yuji pulled off some tensor black magic

04/02 (Yuji)
* Integrated pointer net into QuACC
