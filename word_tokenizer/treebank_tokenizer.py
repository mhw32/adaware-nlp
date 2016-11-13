'''
TreeBank Tokenization:

- most punctuation is split from adjoining words.
- double quotes (") are changed to doubled single forward-
    and backward- quotes (`` and '')
- verb contractions and the Anglo-Saxon genitive of nouns are split
    into their component morphemes, and each morpheme is tagged separately.

    Edge Cases
    ----------
    children's --> children 's
    parents' --> parents '
    won't --> wo n't
    gonna --> gon na
    I'm --> I 'm

    Examples
    --------

    "Good muffins cost $3.88 in New York"
    ["Good", "muffins", "cost", "$", "3.88", "in", "New", "York"]

    "Please buy me two of them, thanks."
    ["Please", "buy", "me", "two", "of", "them,", "thanks", "."]

    "I've had a number of interesting conversations"
    ["I", "'ve", "had", "a", "number", "of", "interesting", "conversations"]

    "He's an ex--Intel engineer"
    ["He", "'s", "an", "ex", "--", "Intel", "engineer"]

This tokenization allows us to analyze each component separately,
so (for example) "I" can be in the subject Noun Phrase while "'m" is the
head of the main verb phrase.

There are some subtleties for hyphens vs. dashes, elipsis dots (...) and
so on, but these often depend on the particular corpus or application
of the tagged data.

In parsed corpora, bracket-like characters are converted to special 3-letter sequences,
to avoid confusion with parse brackets. Some POS taggers, such as Adwait Ratnaparkhi's
MXPOST, require this form for their input.

In other words, these tokens in POS files: ( ) [ ] { } become, in parsed files:
-LRB- -RRB- -RSB- -RSB- -LCB- -RCB-
(The acronyms stand for (Left|Right) (Round|Square|Curly) Bracket.)

Based on Sed script by Robert McIntyre
http://www.cis.upenn.edu/~treebank/tokenizer.sed

'''

import re

class TreeBankTokenizer(object):
    ''' Penn Treebank tokenization on arbitrary raw sentence
    '''

    def __init__(self, mxpost=False):
        ''' Args
            ----
            mxpost : bool (default False)
                     turn to True if you want ([{}]) to be turned into
                     LRB/RRB... notation
        '''
        self.mxpost = mxpost

        # starting quotes
        self.STARTING_QUOTES = [
            (re.compile(r'^\"'), r'``'),
            (re.compile(r'(``)'), r' \1 '),
            (re.compile(r'([ (\[{<])"'), r'\1 `` '),
        ]

        # punctuation
        self.PUNCTUATION = [
            (re.compile(r'([:,])([^\d])'), r' \1 \2'),
            (re.compile(r'([:,])$'), r' \1 '),
            (re.compile(r'\.\.\.'), r' ... '),
            (re.compile(r'[;@#$%&]'), r' \g<0> '),
            (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
            (re.compile(r'[?!]'), r' \g<0> '),
            (re.compile(r"([^'])' "), r"\1 ' "),
        ]

        # parens, brackets, etc.
        self.PARENS_BRACKETS = [
            (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> '),
            (re.compile(r'--'), r' -- '),
        ]

        self.MXPOST_PARENS = [
            (re.compile(r'\('), '-LRB-'),
            (re.compile(r'\)'), '-RRB-'),
            (re.compile(r'\['), '-LSB-'),
            (re.compile(r'\]'), '-RSB-'),
            (re.compile(r'\{'), '-LCB-'),
            (re.compile(r'\}'), '-RCB-')
        ]

        # ending quotes
        self.ENDING_QUOTES = [
            (re.compile(r'"'), " '' "),
            (re.compile(r'(\S)(\'\')'), r'\1 \2 '),
            (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
            (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
        ]

        # List of contractions adapted from Robert MacIntyre's tokenizer.
        self.CONTRACTIONS2 = [
            re.compile(r"(?i)\b(can)(not)\b"),
            re.compile(r"(?i)\b(d)('ye)\b"),
            re.compile(r"(?i)\b(gim)(me)\b"),
            re.compile(r"(?i)\b(gon)(na)\b"),
            re.compile(r"(?i)\b(got)(ta)\b"),
            re.compile(r"(?i)\b(lem)(me)\b"),
            re.compile(r"(?i)\b(mor)('n)\b"),
            re.compile(r"(?i)\b(wan)(na) ")]

        self.CONTRACTIONS3 = [
            re.compile(r"(?i) ('t)(is)\b"),
            re.compile(r"(?i) ('t)(was)\b")]

        self.CONTRACTIONS4 = [
            re.compile(r"(?i)\b(whad)(dd)(ya)\b"),
            re.compile(r"(?i)\b(wha)(t)(cha)\b")]

    def tokenize(self, sentence):
        ''' As input, pass a single sentence as a time. This means that
            sentence disambiguation must be completed prior.
        '''

        for reg, sub in self.STARTING_QUOTES:
            sentence = reg.sub(sub, sentence)

        for reg, sub in self.PUNCTUATION:
            sentence = reg.sub(sub, sentence)

        for reg, sub in self.PARENS_BRACKETS:
            sentence = reg.sub(sub, sentence)

        if self.mxpost:
            for reg, sub in self.MXPOST_PARENS:
                sentence = reg.sub(sub, sentence)

        # add extra space to make things easier
        sentence = " " + sentence + " "

        for reg, sub in self.ENDING_QUOTES:
            sentence = reg.sub(sub, sentence)

        for reg in self.CONTRACTIONS2:
            sentence = reg.sub(r' \1 \2 ', sentence)

        for reg in self.CONTRACTIONS3:
            sentence = reg.sub(r' \1 \2 ', sentence)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for reg in self.CONTRACTIONS4:
        #     sentence = reg.sub(r' \1 \2 \3 ', sentence)

        return sentence.split()
