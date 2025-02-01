DOT='.'
SLASH='/'

# MOSES
MOSES_SCRIPTS='mosesdecoder/scripts'
TOKENIZER=MOSES_SCRIPTS + SLASH + 'tokenizer/tokenizer.perl'
CLEAN=MOSES_SCRIPTS+ SLASH +'training/clean-corpus-n.perl'
NORMALIZE_PUNCTUATION=MOSES_SCRIPTS+SLASH+'tokenizer/normalize-punctuation.perl'
RM_NON_PRINTING_CHAR=MOSES_SCRIPTS+SLASH+'tokenizer/remove-non-printing-char.perl'

# BPE
BPE_ROOT='subword-nmt/subword_nmt'
BPE_TOKENS=40000
BPE_CODES=''

# LANGUAGE
SRC, TGT='en', 'ar'
LANG='' # it should equal src-tgt

# DATA
DATAPATH='data_origin'
DATABIN='data_dest'
BPE_OUTPUT=DATABIN # it was named prep in shell file
tmp=DATABIN+SLASH+'tmp'

TRAIN='train'
VALID='valid'
TEST='test'
