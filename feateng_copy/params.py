
# Jordan Boyd_Graber
# 2023
#
# File to define default command line parameters and to instantiate objects
# based on those parameters.  Included in most files in this project.

from pandas import read_csv
import logging
import json
import os



def add_general_params(parser):
    parser.add_argument('--logging_level', type=int, default=logging.INFO)
    parser.add_argument('--logging_file', type=str, default='qanta.log')

def add_question_params(parser):
    parser.add_argument('--limit', type=int, default=1250)
    parser.add_argument('--question_source', type=str, default='json')
    parser.add_argument('--questions', default = "./qanta.buzztrain.json",type=str)
    parser.add_argument('--expo_output_root', default="expo/expo", type=str)
    parser.add_argument('--expo_questions', type=str)

def add_buzzer_params(parser):
    parser.add_argument('--buzzer_guessers', nargs='+', default = ['GprGuesser'], help='Guessers to feed into Buzzer', type=str)
    parser.add_argument('--features', nargs='+', help='Features to feed into Buzzer', type=str,  default=["isEmpty", "inQuestion", "Length", "Expected_Length", "Answer_Sim"])    
    parser.add_argument('--buzzer_type', type=str, default="LogisticBuzzer")
    parser.add_argument('--run_length', type=int, default=100)
    parser.add_argument('--LogisticBuzzer_filename', type=str, default="models/LogisticBuzzer")    
    
def add_guesser_params(parser):
    parser.add_argument('--guesser_type', type=str, default="GprGuesser")
    parser.add_argument('--tfidf_min_length', type=int, help="How long (in characters) must text be before it is indexed?", default=50)
    parser.add_argument('--tfidf_max_length', type=int, help="How long (in characters) must text be to be removed?", default=500)    
    parser.add_argument('--tfidf_split_sentence', type=bool, default=True, help="Index sentences rather than paragraphs")
    parser.add_argument('--wiki_min_frequency', type=int, help="How often must wiki page be an answer before it is used", default=10)
    parser.add_argument('--guesser_answer_field', type=str, default="page", help="Where is the cannonical answer")
    parser.add_argument('--TfidfGuesser_filename', type=str, default="models/TfidfGuesser")
    parser.add_argument('--WikiGuesser_filename', type=str, default="models/WikiGuesser")    
    parser.add_argument('--GprGuesser_filename', type=str, default="models/GprGuesser")
    parser.add_argument('--wiki_zim_filename', type=str, default="data/wikipedia.zim")
    

def setup_logging(flags):
    logging.basicConfig(level=flags.logging_level, force=True)
    
def load_questions(flags):
    questions = None
    if flags.question_source == 'json':
        with open(flags.questions) as infile:
            questions = json.load(infile)
            
    if flags.question_source == 'csv':
        questions = read_csv(flags.questions)

    if flags.question_source == 'expo':
        questions = ExpoWriter(flags.expo_output_root)
        if flags.expo_questions is None:
            questions.debug()
        else:
            questions.load_questions(flags.expo_questions)

        return questions
        
    assert questions is not None, "Did not load %s of type %s" % (flags.questions, flags.question_source)

    if flags.limit > 0:
        questions = questions[:flags.limit]

    logging.info("Read %i questions" % len(questions))
        
    return questions

def instantiate_guesser(guesser_type, flags):
    from gpr_guesser import GprGuesser
    

    guesser = None
    print(guesser_type)
    if guesser_type == "GprGuesser":
        print("HERE")
        guesser = GprGuesser(flags.GprGuesser_filename)
    
        
    assert guesser is not None, "Guesser (type=%s) not initialized" % flags.guesser_type

    return guesser

def load_guesser(flags, load=False):
    """
    Given command line flags, load a guesser
    """

    return instantiate_guesser(flags.guesser_type, flags)

def load_buzzer(flags):
    """
    Create the buzzer and its features.
    """
    
    print("Loading buzzer")
    buzzer = None
    if flags.buzzer_type == "LogisticBuzzer":
        from logistic_buzzer import LogisticBuzzer
        buzzer = LogisticBuzzer(flags.LogisticBuzzer_filename, flags.run_length)
    assert buzzer is not None, "Buzzer (type=%s) not initialized" % flags.buzzer_type


    for gg in flags.buzzer_guessers:
        guesser = instantiate_guesser(gg, flags)
        guesser.load()
        logging.info("Adding %s to Buzzer" % gg)
        buzzer.add_guesser(gg, guesser, gg==flags.guesser_type)

    print("Initializing features: %s" % str(flags.features))
    print("dataset: %s" % str(flags.questions))

    ######################################################################
    ######################################################################
    ######################################################################
    ######
    ######
    ######  For the feature engineering homework, here's where you need
    ######  to add your features to the buzzer.
    ######
    ######
    ######################################################################
    ######################################################################
    ######################################################################    
    
    for ff in flags.features:
        if ff == "Length":
            from features import LengthFeature
            feature = LengthFeature(ff)
            buzzer.add_feature(feature)

        if ff == "Frequency":                                  
            from features import FrequencyFeature              
            feature = FrequencyFeature(ff)                     
            buzzer.add_feature(feature)
        
        if ff == "FinalPOS":
            from features import FinalPOS
            feature = FinalPOS(ff)
            buzzer.add_feature(feature)
        
        if ff == "Expected_Length":
            from features import Expected_Length
            feature = Expected_Length(ff)
            feature.add_training("../data/qanta.buzztrain.json")
            buzzer.add_feature(feature)

        if ff == "Category_Length":
            from features import Category_Length
            feature = Category_Length(ff)
            feature.add_training("../data/qanta.buzztrain.json")
            buzzer.add_feature(feature)

        if ff == "Answer_Sim":
            from features import Answer_Similarity
            feature = Answer_Similarity(ff)
            feature.get_data()
            buzzer.add_feature(feature)

        if ff == "isEmpty":
            from features import isEmpty
            feature = isEmpty(ff)
            buzzer.add_feature(feature)

        if ff == "inQuestion":
            from features import inQuestion
            feature = inQuestion(ff)
            buzzer.add_feature(feature)
    
    return buzzer
