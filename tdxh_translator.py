# _*_ coding:utf-8 _*_
# MBartTranslator :
# Author : ParisNeo https://github.com/ParisNeo/prompt_translator.git
# Description : This script translates Stable diffusion prompt from one of the 50 languages supported by MBART
#    It uses MBartTranslator class that provides a simple interface for translating text using the MBart language model.

# pip install sentencepiece


# lib\site-packages、transformers\generation\utils.py:1273: UserWarning: Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to 200 (`generation_config.max_length`). Controlling `max_length` via the config is deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
# to aviod this warnings,add 
# /mbart-large-50-many-to-many-mmt/config
# "max_new_tokens":200,


from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re
import os

my_dir = os.path.dirname(os.path.abspath(__file__))
# The directory to store the models
model_mbart = os.path.join(my_dir, "model", "mbart-large-50-many-to-many-mmt__only_to_English")
# cache_dir = "model"

class MBartTranslator:
    """MBartTranslator class provides a simple interface for translating text using the MBart language model.

    The class can translate between 50 languages and is based on the "facebook/mbart-large-50-many-to-many-mmt"
    pre-trained MBart model. However, it is possible to use a different MBart model by specifying its name.

    Attributes:
        model (MBartForConditionalGeneration): The MBart language model.
        tokenizer (MBart50TokenizerFast): The MBart tokenizer.
    """

    def __init__(self, model_mbart=model_mbart, src_lang=None, tgt_lang=None):
        self.supported_languages = [
            "ar_AR",
            "de_DE",
            "en_XX",
            "es_XX",
            "fr_XX",
            "hi_IN",
            "it_IT",
            "ja_XX",
            "ko_XX",
            "pt_XX",
            "ru_RU",
            "zh_XX",
            "af_ZA",
            "bn_BD",
            "bs_XX",
            "ca_XX",
            "cs_CZ",
            "da_XX",
            "el_GR",
            "et_EE",
            "fa_IR",
            "fi_FI",
            "gu_IN",
            "he_IL",
            "hi_XX",
            "hr_HR",
            "hu_HU",
            "id_ID",
            "is_IS",
            "ja_XX",
            "jv_XX",
            "ka_GE",
            "kk_XX",
            "km_KH",
            "kn_IN",
            "ko_KR",
            "lo_LA",
            "lt_LT",
            "lv_LV",
            "mk_MK",
            "ml_IN",
            "mr_IN",
            "ms_MY",
            "ne_NP",
            "nl_XX",
            "no_XX",
            "pl_XX",
            "ro_RO",
            "si_LK",
            "sk_SK",
            "sl_SI",
            "sq_AL",
            "sr_XX",
            "sv_XX",
            "sw_TZ",
            "ta_IN",
            "te_IN",
            "th_TH",
            "tl_PH",
            "tr_TR",
            "uk_UA",
            "ur_PK",
            "vi_VN",
            "war_PH",
            "yue_XX",
            "zh_CN",
            "zh_TW",
        ]
        print("Building translator")
        # print("Loading generator (this may take few minutes the first time as I need to download the model)")
        print("Loading generator")
        # self.model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = MBartForConditionalGeneration.from_pretrained(model_mbart)
        print("Loading tokenizer")
        # self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang, cache_dir=cache_dir)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_mbart, src_lang=src_lang, tgt_lang=tgt_lang)
        print("Translator is ready")

    def translate(self, text: str, input_language: str, output_language: str) -> str:
        """Translate the given text from the input language to the output language.

        Args:
            text (str): The text to translate.
            input_language (str): The input language code (e.g. "hi_IN" for Hindi).
            output_language (str): The output language code (e.g. "en_US" for English).

        Returns:
            str: The translated text.
        """
        if input_language not in self.supported_languages:
            raise ValueError(f"Input language not supported. Supported languages: {self.supported_languages}")
        if output_language not in self.supported_languages:
            raise ValueError(f"Output language not supported. Supported languages: {self.supported_languages}")

        self.tokenizer.src_lang = input_language
        encoded_input = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_input, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id[output_language]
        )
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return translated_text[0]

class LanguageOption:
    """
    A class representing a language option in a language selector.

    Attributes:
        label (str): The display label for the language option.
        code (str): The ISO 639-1 language code for the language option.
    """

    def __init__(self, label, code):
        """
        Initializes a new LanguageOption instance.

        Args:
            label (str): The display label for the language option.
            code (str): The ISO 639-1 language code for the language option.
        """
        self.label = label
        self.code = code

'''
# This is a list of LanguageOption objects that represent the various language options available.
# Each LanguageOption object contains a label that represents the display name of the language and 
# a language code that represents the code for the language that will be used by the translation model.
# The language codes follow a format of "xx_YY" where "xx" represents the language code and "YY" represents the 
# country or region code. If the language code is not specific to a country or region, then "XX" is used instead.
# For example, "en_XX" represents English language and "fr_FR" represents French language specific to France.
# These LanguageOption objects will be used to display the language options to the user and to retrieve the 
# corresponding language code when the user selects a language.
'''

language_options = [
    LanguageOption("عربية", "ar_AR"),
    LanguageOption("Deutsch", "de_DE"),
    LanguageOption("Español", "es_XX"),
    LanguageOption("Français", "fr_XX"),
    LanguageOption("हिन्दी", "hi_IN"),
    LanguageOption("Italiano", "it_IT"),
    LanguageOption("日本語", "ja_XX"),
    LanguageOption("한국어", "ko_XX"),
    LanguageOption("Português", "pt_XX"),
    LanguageOption("Русский", "ru_RU"),
    LanguageOption("中文", "zh_CN"),
    LanguageOption("Afrikaans", "af_ZA"),
    LanguageOption("বাংলা", "bn_BD"),
    LanguageOption("Bosanski", "bs_XX"),
    LanguageOption("Català", "ca_XX"),
    LanguageOption("Čeština", "cs_CZ"),
    LanguageOption("Dansk", "da_XX"),
    LanguageOption("Ελληνικά", "el_GR"),
    LanguageOption("Eesti", "et_EE"),
    LanguageOption("فارسی", "fa_IR"),
    LanguageOption("Suomi", "fi_FI"),
    LanguageOption("ગુજરાતી", "gu_IN"),
    LanguageOption("עברית", "he_IL"),
    LanguageOption("हिन्दी", "hi_XX"),
    LanguageOption("Hrvatski", "hr_HR"),
    LanguageOption("Magyar", "hu_HU"),
    LanguageOption("Bahasa Indonesia", "id_ID"),
    LanguageOption("Íslenska", "is_IS"),
    LanguageOption("Javanese", "jv_XX"),
    LanguageOption("ქართული", "ka_GE"),
    LanguageOption("Қазақ", "kk_XX"),
    LanguageOption("ខ្មែរ", "km_KH"),
    LanguageOption("ಕನ್ನಡ", "kn_IN"),
    LanguageOption("한국어", "ko_KR"),
    LanguageOption("ລາວ", "lo_LA"),
    LanguageOption("Lietuvių", "lt_LT"),
    LanguageOption("Latviešu", "lv_LV"),
    LanguageOption("Македонски", "mk_MK"),
    LanguageOption("മലയാളം", "ml_IN"),
    LanguageOption("मराठी", "mr_IN"),
    LanguageOption("Bahasa Melayu", "ms_MY"),
    LanguageOption("नेपाली", "ne_NP"),
    LanguageOption("Nederlands", "nl_XX"),
    LanguageOption("Norsk", "no_XX"),
    LanguageOption("Polski", "pl_XX"),
    LanguageOption("Română", "ro_RO"),
    LanguageOption("සිංහල", "si_LK"),
    LanguageOption("Slovenčina", "sk_SK"),
    LanguageOption("Slovenščina", "sl_SI"),
    LanguageOption("Shqip", "sq_AL"),   
    LanguageOption("Turkish", "tr_TR"),
    LanguageOption("Tiếng Việt", "vi_VN"),
    LanguageOption("English", "en_XX"),
]

def get_language_option(input_label):
    for language_option in language_options:
        if language_option.label == input_label:
            return language_option
    return None  

def remove_unnecessary_spaces(text):
    """Removes unnecessary spaces between characters."""
    pattern = r"\)\s*\+\+|\)\+\+\s*"
    replacement = r")++"
    return re.sub(pattern, replacement, text)

def correct_translation_format(original_text, translated_text):
    original_parts = original_text.split('++')
    translated_parts = translated_text.split('++')
    
    corrected_parts = []
    for i, original_part in enumerate(original_parts):
        translated_part = translated_parts[i]
        
        original_plus_count = original_part.count('+')
        translated_plus_count = translated_part.count('+')
        plus_difference = translated_plus_count - original_plus_count
        
        if plus_difference > 0:
            translated_part = translated_part.replace('+' * plus_difference, '', 1)
        elif plus_difference < 0:
            translated_part += '+' * abs(plus_difference)
        
        corrected_parts.append(translated_part)
    
    corrected_text = '++'.join(corrected_parts)
    return corrected_text

def extract_plus_positions(text):
    """
    Given a string of text, extracts the positions of all sequences of one or more '+' characters.
    
    Args:
    - text (str): the input text
    
    Returns:
    - positions (list of lists): a list of [start, end, count] for each match, where start is the index of the
      first '+' character, end is the index of the last '+' character + 1, and count is the number of '+' characters
      in the match.
    """
    # Match any sequence of one or more '+' characters
    pattern = re.compile(r'\++')

    # Find all matches of the pattern in the text
    matches = pattern.finditer(text)

    # Loop through the matches and add their positions to the output list
    positions = []
    last_match_end = None
    for match in matches:
        if last_match_end is not None and match.start() != last_match_end:
            # If there is a gap between the current match and the previous one, add a new position
            j = last_match_end - 1
            while text[j] == "+":
                j -= 1
            j += 1
            positions.append([j, last_match_end, last_match_end - j])

        last_match_end = match.end()
    
    # If the final match extends to the end of the string, add its position to the output list
    if last_match_end is not None and last_match_end == len(text):
        j = last_match_end - 1
        while text[j] == "+":
            j -= 1
        j += 1
        positions.append([j, last_match_end, last_match_end - j])

    return positions

def match_pluses(original_text, translated_text):
    """
    Given two strings of text, replaces sequences of '+' characters in the second string with the corresponding
    sequences of '+' characters in the first string.
    
    Args:
    - original_text (str): the original text
    - translated_text (str): the translated text with '+' characters
    
    Returns:
    - output (str): the translated text with '+' characters replaced by those in the original text
    """
    in_positions = extract_plus_positions(original_text)
    out_positions = extract_plus_positions(translated_text)    
    
    out_vals = []
    out_current_pos = 0
    
    if len(in_positions) == len(out_positions):
        # Iterate through the positions and replace the sequences of '+' characters in the translated text
        # with those in the original text
        for in_, out_ in zip(in_positions, out_positions):
            out_vals.append(translated_text[out_current_pos:out_[0]])
            out_vals.append(original_text[in_[0]:in_[1]])
            out_current_pos = out_[1]
            
            # Check that the number of '+' characters in the original and translated sequences is the same
            if in_[2] != out_[2]:
                print("detected different + count")

    # Add any remaining text from the translated string to the output
    out_vals.append(translated_text[out_current_pos:])
    
    # Join the output values into a single string
    output = "".join(out_vals)
    return output

def post_process_prompt(original, translated):
    """Applies post-processing to the translated prompt such as removing unnecessary spaces and extra plus signs."""
    clean_prompt = remove_unnecessary_spaces(translated)
    clean_prompt = match_pluses(original, clean_prompt)
    #clean_prompt = remove_extra_plus(clean_prompt)
    return clean_prompt  

class Prompt:
    def __init__(self, positive_prompt_list, negative_prompt_list):
        self.positive_prompt_list = positive_prompt_list
        self.negative_prompt_list = negative_prompt_list

class TranslatorScript:
    def __init__(self) -> None:
        """Initializes the class and sets the default value for enable_translation attribute."""
        self.is_active=False

    def title(self):
        """Returns the title of the class."""
        return "Translate prompt to english"
    
    def set_active(self):
        """Sets the is_active attribute and initializes the translator object if not already created. """
        self.is_active=True
        if not hasattr(self, "translator"):
            self.translator = MBartTranslator()
        return "translator active"  

    def set_deactive(self):
        self.is_active=False
        if hasattr(self, "translator"):
            del self.translator
        return "translator deactive"
    
    def tranlate(self,original_list,original_language,target_language):
        if original_list ==[]:
            return [""]
        if original_list is None:
            return [""]
        translated_list=[]
        previous = ""
        previous_translated = ""
        for original in original_list:
            if previous != original:
                original_language_option = get_language_option(original_language)
                if original_language_option == None:
                    print("The original language label may be wrong")
                    return None
                target_language_option = get_language_option(target_language)
                if target_language_option == None:
                    print("The target language label may be wrong")
                    return None
                
                print(f"------Translating words to {target_language_option.label} from {original_language_option.label}")
                print(f"Initial words:{original}")

                translated = self.translator.translate(original, original_language_option.code, target_language_option.code) 
                
                translated = post_process_prompt(original, translated)

                print(f"Translated words:{translated}")
                translated_list.append(translated)

                previous=original
                previous_translated = translated
            else:
                translated_list.append(previous_translated)
        return translated_list
        
    def process(self, p_in:Prompt,original_language, target_language="English"):
        """Translates the words from original_language to target_language using the MBartTranslator object."""
        if not hasattr(self, "translator") or not self.is_active:
            print("Please set the translator active.")
            return 
        p_in.positive_prompt_list= self.tranlate(p_in.positive_prompt_list,original_language,target_language)
        p_in.negative_prompt_list= self.tranlate(p_in.negative_prompt_list,original_language,target_language)

if __name__ == "__main__":
    positive_prompt_list=["一个女孩，\n在海边，跳舞，","衣袂飘飘。"]
    negative_prompt_list=["油画，二次元","树木，太阳伞，丑陋。"]
    p_in = Prompt(positive_prompt_list, negative_prompt_list)

    translator = TranslatorScript()
    translator.set_active()
    translator.process(p_in,"中文")
    translator.set_deactive()
    print(p_in.positive_prompt_list)
    print(p_in.negative_prompt_list)

    


