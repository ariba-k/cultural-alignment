from scipy.spatial.distance import jensenshannon
import ast
def calculate_similarity(model_probs, country_probs):
    return 1 - jensenshannon(model_probs, country_probs)

def parse_dict_from_string(s):
    dict_str = s[s.index('(') + 1:s.rindex(')')].split(',', 1)[1].strip()
    dict_converted = ast.literal_eval(dict_str)
    return dict_converted
