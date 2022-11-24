import copy
from hashlib import md5
import json
import os
from modules import sd_samplers, shared, scripts, script_callbacks
from modules.script_callbacks import ImageSaveParams
import modules.images as images
from modules.processing import Processed, process_images, StableDiffusionProcessing
from modules.shared import opts, OptionInfo
from modules.paths import script_path

import gradio as gr
from pathlib import Path
import torch
import torch.nn as nn
import clip
import platform
from launch import is_installed, run_pip
from modules.generation_parameters_copypaste import parse_generation_parameters
import math
import csv
import random
from logging import PlaceHolder
from dataclasses import dataclass, field
import collections
from copy import deepcopy
from scipy.cluster.hierarchy import weighted


# This is some mega janky code in general that is in bad need of rewriting...
# Please don't look too hard at it.

PARAM_NAMES = ["steps_param", "cfg_param", "denoise_param", "samplers_param"]

extension_name = "Aesthetic Image Scorer"
if platform.system() == "Windows" and not is_installed("pywin32"):
    run_pip(f"install pywin32", "pywin32")
try:
    from tools.add_tags import tag_files
except:
    print(f"{extension_name}: Unable to load Windows tagging script from tools directory")
    tag_files = None

state_name = "sac+logos+ava1-l14-linearMSE.pth"
if not Path(state_name).exists():
    url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{state_name}?raw=true"
    import requests
    r = requests.get(url)
    with open(state_name, "wb") as f:
        f.write(r.content)


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


try:
    force_cpu = opts.ais_force_cpu
except:
    force_cpu = False

if force_cpu:
    print(f"{extension_name}: Forcing prediction model to run on CPU")
device = "cuda" if not force_cpu and torch.cuda.is_available() else "cpu"

# load the model you trained previously or the model available in this repo
pt_state = torch.load(state_name, map_location=torch.device(device=device))

# CLIP embedding dim is 768 for CLIP ViT L 14
predictor = AestheticPredictor(768)
predictor.load_state_dict(pt_state)
predictor.to(device)
predictor.eval()

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)


def get_image_features(image, device=device, model=clip_model, preprocess=clip_preprocess):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # l2 normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    return image_features


def get_score(image):
    image_features = get_image_features(image)
    score = predictor(torch.from_numpy(image_features).to(device).float())
    return score.item()

class AISGroup:
    def __init__(self, name="", apply_choices=lambda choice, choice_values: {"tags": [f"{choice}_{choice_values[choice]}"]}, default=[]):
        gen_params = lambda p: {
            "steps": p.steps,
            "sampler": sd_samplers.samplers[p.sampler_index].name,
            "cfg_scale": p.cfg_scale,
            "seed": p.seed,
            "width": p.width,
            "height": p.height,
            "model": shared.sd_model.sd_model_hash,
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
        }
        self.choice_processors = {
            "aesthetic_score": lambda params: params.pnginfo["aesthetic_score"] if "aesthetic_score" in params.pnginfo else round(get_score(params.image), 1),
            "sampler": lambda params: sd_samplers.samplers[params.p.sampler_index].name if params.p is not None and params.p.sampler else None,
            "cfg_scale": lambda params: params.p.cfg_scale if params.p is not None and params.p.cfg_scale else None,
            "sd_model_hash": lambda params: shared.sd_model.sd_model_hash,
            "seed": lambda params: str(int(params.p.seed)) if params.p is not None and params.p.seed else None,
            "hash": lambda params: md5(json.dumps(gen_params(params.p)).encode('utf-8')).hexdigest() if params.p is not None else None,
        }
        self.name = name
        self.apply_choices = apply_choices
        self.default = default

    def get_choices(self):
        return list(self.choice_processors.keys())

    def get_name(self):
        return self.name

    def selected(self, opts):
        return opts.__getattr__(self.name)

    def get_choice_processors(self):
        return self.choice_processors

    def apply(self, choice_values, applied, opts):
        for choice in self.get_choices():
            if choice in self.selected(opts) and choice in choice_values:
                applied_choices = self.apply_choices(choice, choice_values)
                for key, value in applied_choices.items():
                    if key not in applied:
                        applied[key] = self.get_default()
                    if isinstance(applied[key], dict):
                        applied[key].update(value)
                    else:
                        applied[key] = applied[key] + value
        return applied
    
    def get_default(self):
        return self.default 

class AISGroups:
    def __init__(self, groups=[]):
        self.groups = groups
        self.choices = {}
        self.choice_processors = {}
        for group in groups:
            self.choice_processors.update(group.get_choice_processors())
        self.choices = list(self.choice_processors.keys())

    def all_selected(self, opts):
        selected = {}
        for group in self.groups:
            for select in group.selected(opts):
                selected.update({select: 0})
        return selected.keys()

    def apply(self, opts: list, params: ImageSaveParams):
        parsed_info = parse_generation_parameters(
            params.pnginfo.get("parameters", ""))
        params_hash = md5(json.dumps(
            str(vars(params)), default=lambda o: dir(o), sort_keys=True).encode('utf-8')).hexdigest()
        applied = {}
        if params_hash not in output_cache:
            choice_values = {}
            choices_selected = self.all_selected(opts)
            for choice, processor in self.choice_processors.items():
                if choice in choices_selected:
                    choice_values.update({choice: processor(params)})

            if "seed" in choice_values and choice_values["seed"] is not None and int(choice_values["seed"]) == -1:
                choice_values["seed"] = int(parsed_info["Seed"]) if "Seed" in parsed_info else int(choice_values["seed"])
            
            for group in self.groups:
                applied = group.apply(choice_values, applied, opts)

            expected_keys = ["tags", "categories", "info", "pnginfo"]
            for key in expected_keys:
                if key not in applied:
                    applied[key] = []
            
            output_cache[params_hash] = applied
        else:
            applied = output_cache[params_hash]
            output_cache.clear()
        
        return applied


ais_exif_pnginfo_choices = AISGroup(name="ais_exif_pnginfo_group", apply_choices=lambda choice, choice_values: {
                                              "pnginfo": {choice: choice_values[choice]}}, default={})
ais_windows_tag_group_choices = AISGroup(name="ais_windows_tag_group")
ais_windows_category_group_choices = AISGroup(name="ais_windows_category_group", apply_choices=lambda choice, choice_values: {
                                              "categories": [f"{choice}_{choice_values[choice]}"]})
ais_generation_params_text_choices = AISGroup(name="ais_generation_params_text_group", apply_choices=lambda choice, choice_values: {
                                              "info": {choice: choice_values[choice]}}, default={})

ais_group = AISGroups([
    ais_windows_tag_group_choices,
    ais_windows_category_group_choices,
    ais_generation_params_text_choices,
    ais_exif_pnginfo_choices,
])

output_cache = {}

def on_ui_settings():
    options = {}

    options.update(shared.options_section(('ais', extension_name), {
        ais_exif_pnginfo_choices.get_name(): OptionInfo([], "Save score as EXIF or PNG Info Chunk", gr.CheckboxGroup, {"choices": ais_exif_pnginfo_choices.get_choices()}),
        ais_windows_tag_group_choices.get_name(): OptionInfo([], "Save tags (Windows only)", gr.CheckboxGroup, {"choices": ais_windows_tag_group_choices.get_choices()}),
        ais_windows_category_group_choices.get_name(): OptionInfo([], "Save category (Windows only)", gr.CheckboxGroup, {"choices": ais_windows_category_group_choices.get_choices()}),
        ais_generation_params_text_choices.get_name(): OptionInfo([], "Save generation params text", gr.CheckboxGroup, {"choices": ais_generation_params_text_choices.get_choices()}),
        "ais_force_cpu": OptionInfo(False, "Force CPU (Requires Custom Script Reload)"),
    }))

    opts.add_option(ais_exif_pnginfo_choices.get_name(),
                    options[ais_exif_pnginfo_choices.get_name()])
    opts.add_option(ais_windows_tag_group_choices.get_name(),
                    options[ais_windows_tag_group_choices.get_name()])
    opts.add_option(ais_windows_category_group_choices.get_name(),
                    options[ais_windows_category_group_choices.get_name()])
    opts.add_option(ais_generation_params_text_choices.get_name(),
                    options[ais_generation_params_text_choices.get_name()])
    opts.add_option("ais_force_cpu", options["ais_force_cpu"])

def on_before_image_saved(params: ImageSaveParams):    
    applied = ais_group.apply(opts, params)
    if len(applied["pnginfo"]) > 0:
        params.pnginfo.update(applied["pnginfo"])
    
    if len(applied["info"]) > 0:
        parts = []
        for label, value in applied["info"].items():
            parts.append(f"{label}: {value}")
        if len(parts) > 0:
            if len(params.pnginfo["parameters"]) > 0:
                params.pnginfo["parameters"] += ", "
            params.pnginfo["parameters"] += f"{', '.join(parts)}\n"
    
    return params
        
def on_image_saved(params: ImageSaveParams):
    filename = os.path.realpath(os.path.join(script_path, params.filename))
    applied = ais_group.apply(opts, params)
    if tag_files is not None:
        tag_files(filename=filename, tags=applied["tags"], categories=applied["categories"],
                    log_prefix=f"{extension_name}: ")
    elif platform.system() == "Windows":
        print(f"{extension_name}: Unable to load tagging script")

def laplace_normalise(data, offset):
    low = min(data)
    return [x+abs(low)+offset for x in data]

def weighted_choice(data, weights, offset = 0, formula = lambda x: x**1.5):
    weights = laplace_normalise(weights, offset)
    weights = [formula(x) for x in weights]
    random_object = random.choices(data, weights)[0]
    return random_object

def prompt_to_list(prompt):
    return prompt.split(",")

def list_to_prompt(prompt):
    if not prompt:
        return prompt
    prompt = ",".join(prompt)
    if prompt[-1] != ",":
        prompt += ","
    return prompt

def dict_to_list_tuple(dict):
    return zip(*dict.items())
    
def add_to_prompt(prompt, weight_dict, offset, seen_list=None):
    prompt_list = prompt_to_list(prompt)[:-1]
    tags, weights = zip(*weight_dict.items())
    seen_list = seen_list + prompt_list
    addables = []
    addables_weights = []
    for i, word in enumerate(tags):
        if word in seen_list:
            seen_list.remove(word)
        else:
            addables.append(word)
            addables_weights.append(weights[i])
    if addables_weights:
        random_tag = weighted_choice(addables, addables_weights, offset)
    else:
        return False
    prompt = list_to_prompt(prompt_list)
    prompt += random_tag + ","
    return (prompt, random_tag)
    
def remove_from_prompt(prompt, weight_dict, offset, seen_list = None):
    prompt_list = prompt_to_list(prompt)[:-1]
    if seen_list:
        seen_list = seen_list[:]
    tags, weights = zip(*weight_dict.items())
    removables = []
    removables_weights = []
    for i, word in enumerate(prompt_list):
        if word in seen_list:
            seen_list.remove(word)
        else:
            removables.append(word)
            removables_weights.append(weights[tags.index(word)])
    if removables_weights:
        random_tag = weighted_choice(removables, removables_weights, offset)
    else:
        return False
    prompt_list.remove(random_tag)
    return (list_to_prompt(prompt_list), random_tag)

def pick_new_param(vals, seen_list = None):
    return random.choice([x for x in vals if x not in seen_list])

@dataclass
class State():
    prompt:             str = ""
    params:             dict = field(default_factory = lambda: {"sampler_index"         : 0,
                                                                "steps"                 : 0,
                                                                "cfg_scale"             : 0,
                                                                "denoising_strength"    : 0
                                                                })
    seed:               int = 0
    visited_prompts:    list = field(default_factory=list) 
    visited_removed:    list = field(default_factory=list) 
    visited_neg:        list = field(default_factory=list) 
    visited_params:     dict = field(default_factory=dict)
    score:              int = 0
    
    def __post_init__(self):
        for key in self.params:
            self.visited_params[key] = [self.params[key]]
    
    def __hash__(self):
        return(hash((self.prompt, self.params["sampler_index"], self.params["steps"], self.params["cfg_scale"], self.params["denoising_strength"])))
    
    def __eq__(self, other):
        if isinstance(other, State):
            if self.prompt == other.prompt and \
                self.params["steps"] == other.params["steps"] and \
                self.params["cfg_scale"] == other.params["cfg_scale"] and \
                self.params["denoising_strength"] == other.params["denoising_strength"] and \
                self.params["sampler_index"] == other.params["sampler_index"]:
                return True
            else:
                return False
        return False
    
    def copy_p_params_to_state(self, p):
        self.params["sampler_index"] = p.sampler_index
        self.params["steps"] = p.steps
        self.params["cfg_scale"] = p.cfg_scale
        self.params["denoising_strength"] = p.denoising_strength
        self.seed = p.seed

    
def convert_state_to_p(state, p):
    p.prompt = state.prompt
    p.steps = state.params["steps"]
    p.cfg_scale = state.params["cfg_scale"]
    p.denoising_strength = state.params["denoising_strength"]
    p.sampler_index = state.params["sampler_index"]
    p.seed = state.seed
    return p


def comma_sep_to_list(c_string):
    return [x.strip() for x in c_string.split(",")]

def comma_sep_string_to_cast_tuple(c_string, f):
    return tuple([f(x) for x in comma_sep_to_list(c_string)])

def clean_string(c_string):
    return ",".join(remove_whitespace_from_list(comma_sep_to_list(c_string)))

def remove_whitespace_from_list(c_list):
    return [x for x in c_list if x != ""]

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
    
class Script(scripts.Script):
    def title(self):
        return "Score optimiser"    

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        use_file        = gr.Checkbox(label = "Use file called prompts.csv", value=True)
        prompt_txt      = gr.Textbox(label = "Comma separated prompt, untick the above box if used", lines=1, placeholder="tag1,tag2,tag3")
        n_steps         = gr.Number(label = 'Steps, total images generated = steps * batch count', value = 40)
        min_improvement = gr.Number(label = 'Minimum score improvement to accept tag', value = 0.05)
        n_patience      = gr.Number(label = "Return to initial prompt if stuck for n steps, 0 = no restart", value=10, precision = 0)
        mem_formula     = gr.Checkbox(label = "Weigh past score effects of applying a tag to future tag applications (only applies to current optimisation batch, can't be turned off yet)", value=True)
        mem_smoothing   = gr.Number(label = "Tag weight offset n > 0, higher = smaller effect of tag weight", value=0.1)
        remove_chance   = gr.Number(label = "Allow random removing instead of adding to prompt with n <= 1 chance", value = 0.3)
        param_chance    = gr.Number(label = "Allow changing one of below sampling parameters instead of prompt with n <= 1 chance", value = 0)
        allow_seed      = gr.Checkbox(label = "Seed randomisation (useful for searching good seeds, leave search string empty above, above two boxes at 0, 1 respectively and below params empty)", value = False)
        steps_param     = gr.Textbox(label = "Sampling steps", placeholder =  "min, max, step")
        cfg_param       = gr.Textbox(label = "CFG Scale", placeholder = "min, max, step")
        denoise_param   = gr.Textbox(label = "Denoise strength", placeholder = "min, max, step")
        samplers_param  = gr.Textbox(label = "Samplers", placeholder = "Not yet implemented")
        allow_punc      = gr.Checkbox(label = "Not yet implemented", value = False)
        return [use_file, prompt_txt,n_steps, min_improvement, n_patience, mem_formula, mem_smoothing, remove_chance, param_chance, allow_seed, steps_param, cfg_param, denoise_param, samplers_param, allow_punc]
    
    
    def run(self, p, use_file, prompt_txt, n_steps, min_improvement, n_patience, mem_formula, mem_smoothing, remove_chance, param_chance, allow_seed, steps_param, cfg_param, denoise_param, samplers_param, allow_punc):
        random_seed = random.randint(0, 2**32-1)
        all_tags = []
        if use_file:
            with open ("prompts.csv", newline="") as f:
                all_tags = [x for line in list(csv.reader(f)) for x in line]
        else:
            all_tags = prompt_txt.split(",")
        all_tags = remove_whitespace_from_list(all_tags)
        init_params = {"steps"              : steps_param,
                      "cfg_scale"           : cfg_param,
                      "denoising_strength"  : denoise_param}
        all_params = {}
        for param, val in init_params.items():
            if val == "":
                continue
            if param in ["steps", "sampler_index"]:
                all_params[param] = comma_sep_string_to_cast_tuple(val, int)
            else:
                all_params[param] = comma_sep_string_to_cast_tuple(val, float)
        cur_prompt = clean_string(p.prompt)
        if cur_prompt[-1] != ",":
            cur_prompt += ","
        images = []
        all_prompts = []
        infotexts = []
        prompt_and_score_and_seed_and_params = []
        stuck_for = 0
        tag = ""
        tag_weights = {x:0 for x in all_tags}
        tag_improvements = {x:0 for x in all_tags}
        remove_weights = {x:0 for x in all_tags + prompt_to_list(cur_prompt)}
        remove_improvements = {x:0 for x in all_tags + prompt_to_list(cur_prompt)}
        mode = "none"
        first_state = State(cur_prompt)
        first_params = {}
        if p.seed == -1:
            p.seed = random_seed
        for param in ["sampler_index", "steps", "cfg_scale", "denoising_strength"]:
            if param in all_params:
                first_params[param] = all_params[param][0]
            else:
                exec(f"first_params[param] = p.{param}")
        first_state = State(cur_prompt, first_params, p.seed)
        p = convert_state_to_p(first_state, p)
        cur_state = first_state
        best_state = first_state
        p.prompt = cur_prompt
        new_best = False
        did_reset = False
        done = False
        overall_best_state = first_state
        for i in range(int(n_steps)):
            n_images = p.n_iter
            cur_score = 0
            processed = process_images(p)
            for image in processed.images:
                score = get_score(image)
                if round(score, 1) == 4.6:
                    n_images -= 1
                else:  
                  cur_score += score
            cur_state.score = 1
            if n_images > 0:
                cur_score /= n_images
                cur_state.score = cur_score
            if i:
                s_diff = cur_state.score - prev_state.score
                if mode == "add":
                    tag_weights[tag] += s_diff
                    if s_diff > 0:
                        tag_improvements[tag] += s_diff
                if mode == "remove":
                    remove_weights[tag] += cur_state.score - prev_state.score
                    if s_diff > 0:
                        remove_improvements[tag] += s_diff
            print(f"\nStep:{i}/{n_steps} \nCurrent score: {cur_state.score} \n Prompt: {cur_state.prompt} \n Params: {cur_state.params} \n Seed: {cur_state.seed}\n")
            prompt_and_score_and_seed_and_params.append((cur_state.prompt,cur_state.score, cur_state.seed,cur_state.params))
            prev_state = cur_state
            if cur_score - min_improvement > best_state.score:
                best_state = cur_state
                stuck_for = 0
                print("New best \n")
                if cur_score - min_improvement > overall_best_state.score:
                    overall_best_state = cur_state
            else:
                cur_prompt = best_state.prompt
                cur_state = best_state
                stuck_for += 1
            images += processed.images
            all_prompts += processed.all_prompts
            infotexts += processed.infotexts
            prompt_length = len(prompt_to_list(cur_prompt))
            param_roll = random.random()
            remove_roll = random.random()
            tried_all = False
            can_do_something = False
            while True:
                if stuck_for > n_patience and n_patience:
                    cur_state = first_state
                    p = convert_state_to_p(cur_state, p)
                    stuck_for = 0
                    did_reset = True
                    best_state = first_state
                    cur_prompt = first_state.prompt
                if param_roll > param_chance:
                    addables = add_to_prompt(cur_prompt, tag_weights, mem_smoothing, cur_state.visited_prompts)
                    if addables:
                        cur_prompt, tag = addables
                        cur_state.visited_prompts.append(tag)
                        mode = "add"
                        can_do_something = True
                        break
                if remove_roll > remove_chance:
                    shuffled_params = copy.deepcopy(list(all_params.items()))
                    n_params = len(shuffled_params)
                    random.shuffle(shuffled_params)
                    new_param_dict = copy.deepcopy(cur_state.params)
                    if allow_seed and random.random() < (1 / (len(all_params) + 1)):
                        cur_state = State(cur_prompt, new_param_dict, random.randint(0, 2**32-1))
                        cur_state.set_visited_defaults
                        p = convert_state_to_p(cur_state, p)
                        mode = "seed"
                        can_do_something = True
                        break
                    else:
                        for i in range(n_params):
                            param, param_interval = shuffled_params.pop()
                            possible_param_vals = frange(*param_interval)
                            if len(possible_param_vals) == len(cur_state.visited_params[param]):
                                continue
                            new_param_val = pick_new_param(possible_param_vals, cur_state.visited_params[param])
                            new_param_dict[param] = new_param_val
                            cur_state.visited_params[param].append(new_param_val)
                            already_visited_param = cur_state.visited_params[param][:]
                            cur_state = State(cur_prompt, new_param_dict, prev_state.seed)
                            cur_state.visited_params[param] = list(set(cur_state.visited_params[param] + already_visited_param))
                            p = convert_state_to_p(cur_state, p)
                            mode = "param"
                            can_do_something = True
                            tag = param
                            break
                        if can_do_something:
                            break
                if prompt_length > 2 and len(cur_state.visited_removed) < prompt_length and remove_chance > 0:
                    removables = remove_from_prompt(cur_prompt, remove_weights, mem_smoothing, cur_state.visited_removed)
                    if removables:
                        cur_prompt, tag = removables
                        cur_state.visited_removed.append(tag)
                        mode = "remove"
                        can_do_something = True
                        break
                if not tried_all:
                    param_roll = 0.999
                    remove_roll = 0.999
                    tried_all = True
                    continue
                if n_patience and not did_reset:
                    stuck_for = n_patience + 1
                    continue
                else:
                    print("ran out of tunable params")
                    done = True
                    break
                break
            if done:
                break
            tried_all = False
            did_reset = False
            if mode not in ["param", "seed"]:
                cur_state = State(cur_prompt, prev_state.params, prev_state.seed)
                if mode == "add":
                    cur_state.visited_removed.append(tag)
                p = convert_state_to_p(cur_state, p)
            
        print("Best score: {} \n Prompt: {}".format(overall_best_state.score, overall_best_state.prompt, overall_best_state.params))
        f_path = find_free_filename("./log/optimiser_log", ".txt")
        with open(f_path, "w") as f:
            for (prompt, score, seed, params) in prompt_and_score_and_seed_and_params:
                f.write("Average score: {}\n Prompt: {}\n Params: seed:{}{}\n".format(score, prompt, seed, params))
                f.write("\n")
            f.write("BEST SCORE:{} \n BEST PROMPT: {} \n BEST PARAMS: seed {} {}\n".format(overall_best_state.score, overall_best_state.prompt, overall_best_state.seed, overall_best_state.params))
            f.write("-------------------------------------------------------------------------\n")
        
        
        to_write = list(zip(tag_weights.keys(), tag_weights.values(), tag_improvements.values()))
        to_write = [(tag, weight, improvement, improvement - weight, improvement / (improvement - weight)) for tag, weight, improvement in to_write]
        write_data_to_csv("./log/optimiser", ".csv", ["tag", "weight", "total improvement", "improvement ratio"], to_write)
        
        to_write_neg = list(zip(remove_weights.keys(), remove_weights.values(), remove_improvements.values()))
        to_write_neg = [(tag, weight, improvement, improvement - weight, improvement / (improvement - weight)) for tag, weight, improvement in to_write_neg]
        write_data_to_csv("./log/optimiser_remove",".csv",["tag", "weight", "total_improvement", "improvement ratio"], to_write_neg)
        
        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
        
def write_data_to_csv(f_path, extension, header, data):
    f_path = find_free_filename(f_path, extension)
    with open(f_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_before_image_saved)
script_callbacks.on_image_saved(on_image_saved)
