import random
import math
import os
import csv


def pick_unique(vals, seen_list = None):
    return random.choice([x for x in vals if x not in seen_list])

def comma_sep_to_list(c_string):
    return [x.strip() for x in c_string.split(",")]

def comma_sep_string_to_cast_tuple(c_string, f):
    return tuple([f(x) for x in comma_sep_to_list(c_string)])

def clean_string(c_string):
    return ",".join(remove_whitespace_from_list(comma_sep_to_list(c_string)))
    
def remove_whitespace_from_list(c_list):
    return [x for x in c_list if x != ""]

def prompt_length(c_string):
    return len(comma_sep_to_list(c_string))

def random_from_dict(dictionary):
    random_key = random.choice(list(dictionary.keys()))
    return (random_key, dictionary[random_key])

def frange(min, max, step):
    n_intervals = (max-min)/step
    if math.isclose(n_intervals, round(n_intervals)):
        n_intervals = round(n_intervals)
    return [x*step+min for x in range(int(n_intervals) + 1)]
    
def find_free_filename(f_path, extension):
    for i in range(10000):
        if not os.path.exists(f"{f_path}{i}{extension}"):
            return f"{f_path}{i}{extension}"
        
def flatten(nested_list):
    flat_list = []
    for list in nested_list:
        flat_list.extend(list)
    return flat_list
    
def laplace_normalise(data, offset):
    low = min(data)
    return [x+abs(low)+offset for x in data]

def weighted_choice(data, weights, offset = 0, formula = lambda x: x**1.5):
    weights = laplace_normalise(weights, offset)
    weights = [formula(x) for x in weights]
    random_object = random.choices(data, weights)[0]
    return random_object

def weighted_choice_multiple(data, weights, n = 2, offset = 0, formula = lambda x: x**1.5):
    weights = laplace_normalise(weights, offset)
    weights = [formula(x) for x in weights]
    random_objects = random.choices(data, weights, k = n)
    return random_objects

def prompt_to_list(prompt):
    return prompt.split(",")

def list_to_prompt(prompt):
    if not prompt:
        return prompt
    return ",".join(prompt)

def dict_to_list_tuple(dict):
    return zip(*dict.items())

def multiply_list(l, n):
    return flatten([[x] * n for x in l])
    
def write_data_to_csv(f_path, extension, header, data):
    f_path = find_free_filename(f_path, extension)
    with open(f_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
